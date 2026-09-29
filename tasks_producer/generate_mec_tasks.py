"""
generate_mec_tasks.py
======================
Generates realistic, DENSE MEC task-arrival CSVs (one per edge server, per
network tier) from the Microsoft GeoLife trajectory dataset, using server
locations and coverage radii defined in `config.py`.

-----------------------------------------------------------------------------
CHANGELOG vs original version (why this rewrite exists)
-----------------------------------------------------------------------------
The original script produced wildly uneven, mostly-empty task counts across
servers within the same tier (one server literally 0 tasks/month, others
under 1 task per 15-min slot, while one or two servers looked dense). Root
causes and fixes:

  BUG/GAP 1 -- Hard coverage cutoff discarded data.
    `assign_server` previously required `d <= coverage_km`, so any point
    outside every server's exact coverage circle contributed nothing. If a
    server's circle happens to sit off the densest GeoLife trajectory
    clusters, it starves completely (this is what happened to the server
    that produced 0 tasks all month). FIX: soft/expanded assignment --
    nearest server within an expanded search radius, falling back to the
    globally nearest server if even that fails. No slot silently discards
    its point anymore. See ASSIGN_EXPANSION_FACTOR.

  GAP 2 -- No per-server density calibration.
    Total task volume per server is driven mostly by how many distinct real
    GeoLife users' trajectories intersect that server's coverage per slot --
    a population effect that a single global BASE_LAMBDA cannot correct for.
    Two servers can be "equally configured" and still differ by 50x in
    volume purely because of where real trajectories happened to cluster.
    FIX: a calibration pre-pass computes each server's total weighted
    "opportunity mass" for the month, then derives a per-server multiplier
    so every server's expected monthly volume converges toward a shared
    TARGET_MONTHLY_TASKS. Relative day/night/weekday/category shape is
    fully preserved -- only the overall scale per server is equalized.

  GAP 3 -- No ambient/background floor.
    Real edge servers always carry some baseline traffic (health checks,
    telemetry, IoT pings) independent of exact human-mobility coverage.
    FIX: AMBIENT_LAMBDA adds a small constant Poisson component to every
    (server, slot), so no slot -- and no whole month -- is structurally
    forced to exactly zero.

  KNOB -- DENSITY_SCALE.
    A single global multiplier on top of everything else, so you can dial
    the whole scenario from "sparse" to "dense offloading" (matching e.g.
    the tasks-1.csv profile: ~15 tasks/15-min-slot, tasks in >98% of slots)
    without touching any of the per-user modeling logic.

CONFUSION: I don't have your actual config.py in this conversation, so the
following assumes it exposes the same interface the original script used:
`cfg.SERVERS` (list of dicts with "id", "lat", "lon", "coverage_km"),
`cfg.SERVER_COUNT[tier]`, `cfg.TIER_6/12/18/30`, `cfg.TASK_PRODUCER_DIR`.
If any field names differ, the mismatch will show up as an ImportError or
KeyError immediately at startup -- easy to spot and fix.
-----------------------------------------------------------------------------
MODELING ASSUMPTIONS (unchanged from original, still documented since the
spec leaves these open)
-----------------------------------------------------------------------------
1. GeoLife trajectories span ~2007-2012 real dates; they do not cover any
   single calendar month. To still produce "one month" of output, each real
   GPS point is reduced to a (day_of_week, 15-min slot_of_day) key. For every
   day in the *synthetic* output month, a user's location for a given slot is
   sampled from whatever real historical points share that same
   (day_of_week, slot) key.

2. Task volume per (user, slot) is modeled as ONE Poisson draw:
       lambda = BASE_LAMBDA * category_mult * time_of_day_mult * weekday_mult
                * server_calibration_mult * DENSITY_SCALE
       n_tasks = Poisson(lambda)
   and priority is a multinomial split of those n_tasks using time-dependent
   probabilities (p_high, p_mid, p_low) that always sum to 1.

3. Users are assigned to one of 5 activity categories (very heavy .. very
   light) deterministically from a seeded RNG, each with its own lambda
   multiplier.

4. Server assignment (Step 2) is done PER TIER, because coverage differs
   with the active server subset. Results are cached per (tier, rounded
   lat/lon) pair since GPS points repeat heavily.

5. A user/slot with no historical GPS coverage for that (day_of_week, slot)
   combination contributes zero *mobility-driven* tasks for that slot, but
   every (server, slot) still receives an ambient background contribution
   (see GAP 3 above), so true structural zeros are eliminated.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, date as date_cls
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# Constants derived from the task spec
# =============================================================================
SLOT_MINUTES = 15
SLOTS_PER_DAY = (24 * 60) // SLOT_MINUTES  # 96
HEADER_LINES_TO_SKIP = 6

TIER_LABELS = {
    cfg.TIER_6: "TIER_6",
    cfg.TIER_12: "TIER_12",
    cfg.TIER_18: "TIER_18",
    cfg.TIER_30: "TIER_30",
}
TIERS_IN_ORDER = [cfg.TIER_6, cfg.TIER_12, cfg.TIER_18, cfg.TIER_30]

# Rounding precision used to de-duplicate / cache GPS points.
COORD_ROUND_DECIMALS = 4

# -----------------------------------------------------------------------
# DENSITY KNOBS -- tune these to move between "sparse" and "dense offloading"
# -----------------------------------------------------------------------
# Base expected tasks per ACTIVE user per 15-minute slot before any
# category / time-of-day / weekday / calibration multipliers are applied.
# BASE_LAMBDA = 0.6

# # Global multiplier applied on top of everything else. Raise this to push
# # the whole scenario denser (e.g. 4-8x gets you into "tasks-1.csv"-like
# # territory: high tasks in almost every slot, double-digit totals per slot
# # during peak hours).
# DENSITY_SCALE = 6.0

# # Target total monthly tasks per server (post-calibration, pre-DENSITY_SCALE
# # and pre-ambient). Each server's raw mobility-driven volume is rescaled
# # toward this target so that servers don't differ by orders of magnitude
# # just because of where real GeoLife trajectories happened to cluster.
# TARGET_MONTHLY_TASKS_PER_SERVER = 3000.0

# # Small constant background lambda added to EVERY (server, slot), regardless
# # of mobility coverage, representing baseline IoT/telemetry/health-check
# # traffic. Sum over a month easily prevents any slot/server from being
# # structurally forced to zero.
# AMBIENT_LAMBDA = 0.35


BASE_LAMBDA = 3.0
DENSITY_SCALE = 6.0
TARGET_MONTHLY_TASKS_PER_SERVER = 10000.0
AMBIENT_LAMBDA = 0.20

# When assigning a GPS point to a server, first try servers within
# coverage_km. If none qualify, expand the search radius by this factor
# before falling back to the single globally-nearest server. This means a
# point is (almost) never discarded outright anymore.
ASSIGN_EXPANSION_FACTOR = 3.0

# User activity categories: (name, lambda multiplier, selection weight)
USER_CATEGORIES = [
    ("very_heavy", 2.6, 0.10),
    ("heavy", 1.7, 0.20),
    ("moderate", 1.0, 0.40),
    ("light", 0.55, 0.20),
    ("very_light", 0.25, 0.10),
]

# =============================================================================
# Time-of-day / weekday multipliers
# =============================================================================
def time_of_day_multiplier(hour: int) -> float:
    """Realistic daily human-activity curve -> Poisson lambda multiplier."""
    if 0 <= hour < 6:
        return 0.15   # midnight / small hours: very low
    if 6 <= hour < 9:
        return 0.60   # morning: moderate
    if 9 <= hour < 17:
        return 1.00   # office hours: high
    if 17 <= hour < 22:
        return 1.30   # evening: highest
    return 0.40       # 22:00-24:00: low


def weekday_multiplier(is_weekend: bool) -> float:
    return 0.85 if is_weekend else 1.00


# def priority_probabilities(hour: int, is_weekend: bool) -> Tuple[float, float, float]:
#     """Returns (p_high, p_mid, p_low), always summing to 1."""
#     if 22 <= hour or hour < 6:
#         return (0.05, 0.15, 0.80)
#     if is_weekend:
#         return (0.15, 0.35, 0.50)
#     if 6 <= hour < 9:
#         return (0.30, 0.40, 0.30)
#     if 9 <= hour < 17:
#         return (0.50, 0.35, 0.15)
#     return (0.20, 0.40, 0.40)

def priority_probabilities(hour: int, is_weekend: bool):
    """Returns (p_high, p_mid, p_low)."""

    # Night: mostly low-priority tasks
    if 22 <= hour or hour < 6:
        return (0.08, 0.22, 0.70)

    # Weekend
    if is_weekend:
        return (0.12, 0.28, 0.60)

    # Morning
    if 6 <= hour < 9:
        return (0.12, 0.28, 0.60)

    # Working hours
    if 9 <= hour < 17:
        return (0.15, 0.30, 0.55)

    # Evening
    return (0.12, 0.28, 0.60)


# =============================================================================
# Haversine distance
# =============================================================================
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


# =============================================================================
# Step 1: Parse user mobility
# =============================================================================
def read_plt_file(filepath: Path):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(HEADER_LINES_TO_SKIP):
                if f.readline() == "":
                    return
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    date_str = parts[5].strip()
                    time_str = parts[6].strip()
                    dt = datetime.strptime(f"{date_str} {time_str}",
                                            "%Y-%m-%d %H:%M:%S")
                except (ValueError, IndexError):
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                yield lat, lon, dt
    except OSError:
        return


def build_user_profile(user_dir: Path) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    traj_dir = user_dir / "Trajectory"
    profile: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    if not traj_dir.is_dir():
        return profile

    seen_this_bucket: Dict[Tuple[int, int], set] = defaultdict(set)

    for plt_file in sorted(traj_dir.glob("*.plt")):
        for lat, lon, dt in read_plt_file(plt_file):
            dow = dt.weekday()
            slot = (dt.hour * 60 + dt.minute) // SLOT_MINUTES
            key = (dow, slot)
            rlat = round(lat, COORD_ROUND_DECIMALS)
            rlon = round(lon, COORD_ROUND_DECIMALS)
            coord = (rlat, rlon)
            if coord in seen_this_bucket[key]:
                continue
            seen_this_bucket[key].add(coord)
            profile[key].append(coord)

    return profile


def read_all_user_profiles(data_dir: Path) -> Dict[str, Dict[Tuple[int, int], List[Tuple[float, float]]]]:
    profiles = {}
    user_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    for user_dir in user_dirs:
        user_id = user_dir.name
        profiles[user_id] = build_user_profile(user_dir)
    return profiles


# =============================================================================
# Step 2: Server assignment (soft / expanded, never fully discards a point)
# =============================================================================
def assign_server(lat: float, lon: float, servers: List[dict],
                   cache: Dict[Tuple[float, float], int]) -> int:
    """
    Nearest-server assignment via Haversine distance. Tries strict coverage
    first; if nothing qualifies, expands the radius by ASSIGN_EXPANSION_FACTOR;
    if STILL nothing qualifies, falls back to the globally nearest server
    regardless of its coverage radius. This guarantees every point is
    assigned to exactly one server -- no data is silently discarded, which
    is what was previously starving/zeroing out some servers.
    """
    key = (lat, lon)
    if key in cache:
        return cache[key]

    best_id, best_dist = None, math.inf
    for srv in servers:
        d = haversine_km(lat, lon, srv["lat"], srv["lon"])
        if d < best_dist:
            best_dist = d
            best_id = srv["id"]
        # track separately whether it's within (expanded) coverage
    # First pass above already finds the globally nearest server as a
    # guaranteed fallback. Now check if a *closer, in-coverage* server
    # exists under strict, then expanded, radius.
    strict_best_id, strict_best_dist = None, math.inf
    expanded_best_id, expanded_best_dist = None, math.inf
    for srv in servers:
        d = haversine_km(lat, lon, srv["lat"], srv["lon"])
        if d <= srv["coverage_km"] and d < strict_best_dist:
            strict_best_dist = d
            strict_best_id = srv["id"]
        if d <= srv["coverage_km"] * ASSIGN_EXPANSION_FACTOR and d < expanded_best_dist:
            expanded_best_dist = d
            expanded_best_id = srv["id"]

    result = strict_best_id if strict_best_id is not None else (
        expanded_best_id if expanded_best_id is not None else best_id)

    cache[key] = result
    return result


# =============================================================================
# User categories (deterministic per user id)
# =============================================================================
def classify_user_category(user_id: str, rng: random.Random) -> Tuple[str, float]:
    names = [c[0] for c in USER_CATEGORIES]
    mults = [c[1] for c in USER_CATEGORIES]
    weights = [c[2] for c in USER_CATEGORIES]
    local_rng = random.Random(f"{SEED}-{user_id}")
    idx = local_rng.choices(range(len(names)), weights=weights, k=1)[0]
    return names[idx], mults[idx]


# =============================================================================
# Step 3 + 4 + 5: time slots, task generation, aggregation
# =============================================================================
def month_dates(start_date: date_cls, num_days: int) -> List[date_cls]:
    return [start_date + timedelta(days=i) for i in range(num_days)]


def _resolve_assignments(tier: int, profiles, dates: List[date_cls], py_rng: random.Random):
    """
    Single pass over every (user, date, slot) that has mobility coverage.
    Resolves the server assignment once per event and records the "weight"
    (category_mult * time_of_day_mult * weekday_mult) that will later drive
    the Poisson lambda. This is used both for calibration (pass 1) and for
    actual generation (pass 2), so location sampling / server assignment is
    identical and deterministic between the two passes.
    """
    n_servers = cfg.SERVER_COUNT[tier]
    servers = cfg.SERVERS[:n_servers]
    assignment_cache: Dict[Tuple[float, float], int] = {}

    events = []  # (server_id, date, slot, weight)
    for user_id, profile in profiles.items():
        if not profile:
            continue

        for d in dates:
            dow = d.weekday()
            is_weekend = dow >= 5
            wk_mult = weekday_multiplier(is_weekend)

            for slot in range(SLOTS_PER_DAY):
                candidates = profile.get((dow, slot))
                if not candidates:
                    continue

                lat, lon = py_rng.choice(candidates)
                server_id = assign_server(lat, lon, servers, assignment_cache)

                hour = (slot * SLOT_MINUTES) // 60
                events.append((user_id, server_id, d, slot, hour, is_weekend, wk_mult))

    return events, [s["id"] for s in servers]


def generate_tier_tasks(tier: int, profiles, user_categories: Dict[str, Tuple[str, float]],
                         dates: List[date_cls], rng: np.random.Generator,
                         py_rng: random.Random) -> Dict[int, Dict[Tuple[date_cls, int], Tuple[int, int, int]]]:
    """
    Returns: server_id -> {(date, slot_index): (high, mid, low)}
    Every (date, slot) combination is present (zero-or-more filled) for every
    server in this tier.
    """
    events, server_ids = _resolve_assignments(tier, profiles, dates, py_rng)

    # ---- Calibration pass: total weighted "opportunity mass" per server ----
    server_weight_totals: Dict[int, float] = {sid: 0.0 for sid in server_ids}
    for user_id, server_id, d, slot, hour, is_weekend, wk_mult in events:
        _, category_mult = user_categories[user_id]
        w = category_mult * time_of_day_multiplier(hour) * wk_mult
        server_weight_totals[server_id] += w

    # calibration multiplier so each server's expected raw (pre-DENSITY_SCALE,
    # pre-ambient) monthly volume converges to TARGET_MONTHLY_TASKS_PER_SERVER
    server_calibration: Dict[int, float] = {}
    for sid in server_ids:
        raw_expected = BASE_LAMBDA * server_weight_totals[sid]
        if raw_expected <= 0:
            # No mobility-driven events reached this server at all. It will
            # still get ambient background traffic; calibration multiplier
            # is irrelevant since there's nothing to scale, so set to 0.
            server_calibration[sid] = 0.0
        else:
            server_calibration[sid] = TARGET_MONTHLY_TASKS_PER_SERVER / raw_expected

    # ---- Generation pass ----
    counts: Dict[int, Dict[Tuple[date_cls, int], List[int]]] = {
        sid: {(d, slot): [0, 0, 0] for d in dates for slot in range(SLOTS_PER_DAY)}
        for sid in server_ids
    }

    for user_id, server_id, d, slot, hour, is_weekend, wk_mult in events:
        _, category_mult = user_categories[user_id]
        lam = (BASE_LAMBDA * category_mult * time_of_day_multiplier(hour) * wk_mult
               * server_calibration[server_id] * DENSITY_SCALE)
        n_tasks = int(rng.poisson(lam)) if lam > 0 else 0
        if n_tasks == 0:
            continue

        p_high, p_mid, p_low = priority_probabilities(hour, is_weekend)
        high, mid, low = rng.multinomial(n_tasks, [p_high, p_mid, p_low])
        bucket = counts[server_id][(d, slot)]
        bucket[0] += int(high)
        bucket[1] += int(mid)
        bucket[2] += int(low)

    # ---- Ambient background floor: applied to every (server, slot) ----
    if AMBIENT_LAMBDA > 0:
        for sid in server_ids:
            for d in dates:
                dow = d.weekday()
                is_weekend = dow >= 5
                for slot in range(SLOTS_PER_DAY):
                    hour = (slot * SLOT_MINUTES) // 60
                    lam = AMBIENT_LAMBDA * time_of_day_multiplier(hour) * weekday_multiplier(is_weekend)
                    n_tasks = int(rng.poisson(lam))
                    if n_tasks == 0:
                        continue
                    p_high, p_mid, p_low = priority_probabilities(hour, is_weekend)
                    high, mid, low = rng.multinomial(n_tasks, [p_high, p_mid, p_low])
                    bucket = counts[sid][(d, slot)]
                    bucket[0] += int(high)
                    bucket[1] += int(mid)
                    bucket[2] += int(low)

    return {sid: {k: tuple(v) for k, v in day_slot_map.items()}
            for sid, day_slot_map in counts.items()}


# =============================================================================
# Step 6: CSV output
# =============================================================================
def write_server_csv(out_path: Path, dates: List[date_cls],
                      slot_counts: Dict[Tuple[date_cls, int], Tuple[int, int, int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "high", "mid", "low"])
        for d in dates:
            for slot in range(SLOTS_PER_DAY):
                minute_of_day = slot * SLOT_MINUTES
                dt = datetime(d.year, d.month, d.day) + timedelta(minutes=minute_of_day)
                high, mid, low = slot_counts[(d, slot)]
                writer.writerow([dt.strftime("%Y-%m-%d %H:%M"), high, mid, low])


# =============================================================================
# Orchestration
# =============================================================================
def main():
    global BASE_LAMBDA, DENSITY_SCALE, TARGET_MONTHLY_TASKS_PER_SERVER, AMBIENT_LAMBDA

    parser = argparse.ArgumentParser(description="Generate dense MEC task-arrival CSVs from GeoLife trajectories.")
    parser.add_argument("--data-dir", type=Path, default=cfg.TASK_PRODUCER_DIR / "Data",
                         help="Root of the GeoLife dataset (contains 000/, 001/, ...)")
    parser.add_argument("--output-dir", type=Path, default=cfg.TASK_PRODUCER_DIR,
                         help="Where to write Tier_X/tasks-N.csv files")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                         help="First day of the synthetic output month (YYYY-MM-DD)")
    parser.add_argument("--num-days", type=int, default=30,
                         help="Number of days to generate (default: 30, i.e. one month)")
    parser.add_argument("--density-scale", type=float, default=DENSITY_SCALE,
                         help="Global multiplier on top of all other lambda factors")
    parser.add_argument("--target-monthly-tasks", type=float, default=TARGET_MONTHLY_TASKS_PER_SERVER,
                         help="Per-server calibration target for mobility-driven monthly volume")
    parser.add_argument("--ambient-lambda", type=float, default=AMBIENT_LAMBDA,
                         help="Background lambda applied to every (server, slot), independent of mobility coverage")
    args = parser.parse_args()

    DENSITY_SCALE = args.density_scale
    TARGET_MONTHLY_TASKS_PER_SERVER = args.target_monthly_tasks
    AMBIENT_LAMBDA = args.ambient_lambda

    random.seed(SEED)
    np.random.seed(SEED)
    py_rng = random.Random(SEED)
    rng = np.random.default_rng(SEED)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    dates = month_dates(start_date, args.num_days)

    print(f"[1/4] Reading trajectories from {args.data_dir} ...")
    profiles = read_all_user_profiles(args.data_dir)
    print(f"      Loaded profiles for {len(profiles)} users.")

    print("[2/4] Classifying user activity categories ...")
    user_categories = {uid: classify_user_category(uid, py_rng) for uid in profiles}

    for tier in TIERS_IN_ORDER:
        label = TIER_LABELS[tier]
        n_servers = cfg.SERVER_COUNT[tier]
        print(f"[3/4] Generating tasks for {label} ({n_servers} servers) "
              f"[density_scale={DENSITY_SCALE}, target/server={TARGET_MONTHLY_TASKS_PER_SERVER}, "
              f"ambient_lambda={AMBIENT_LAMBDA}] ...")

        server_task_counts = generate_tier_tasks(
            tier=tier,
            profiles=profiles,
            user_categories=user_categories,
            dates=dates,
            rng=rng,
            py_rng=py_rng,
        )

        print(f"[4/4] Writing CSVs for {label} ...")
        for i, srv in enumerate(cfg.SERVERS[:n_servers]):
            out_path = args.output_dir / label / f"tasks-{i}.csv"
            write_server_csv(out_path, dates, server_task_counts[srv["id"]])
            total = sum(sum(v) for v in server_task_counts[srv["id"]].values())
            print(f"      server {i} ({srv['id']}): {total} tasks/month")

    print(f"Done. Output written under: {args.output_dir}")


if __name__ == "__main__":
    main()