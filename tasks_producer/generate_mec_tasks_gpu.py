"""
generate_mec_tasks_gpu.py
=========================

GPU-accelerated version of generate_mec_tasks.py.

Main acceleration:
1. Server assignment is vectorized with PyTorch on CUDA.
2. Poisson task generation is vectorized on CUDA instead of calling NumPy
   RNG once per event.
3. Multinomial priority assignment is vectorized on CUDA.
4. Aggregation is performed with torch.index_add_ on CUDA.
5. Falls back automatically to the original CPU-style logic when CUDA is
   unavailable.

The GeoLife text-file parsing and final CSV writing remain CPU operations.

Requirements:
    pip install torch numpy

The script expects the same config.py interface as the original:
    cfg.SERVERS
    cfg.SERVER_COUNT[tier]
    cfg.TIER_6
    cfg.TIER_12
    cfg.TIER_18
    cfg.TIER_30
    cfg.TASK_PRODUCER_DIR
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, date as date_cls
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import config as cfg


# =============================================================================
# Reproducibility
# =============================================================================

SEED = 42

# =============================================================================
# Constants
# =============================================================================

SLOT_MINUTES = 15
SLOTS_PER_DAY = 96
HEADER_LINES_TO_SKIP = 6
COORD_ROUND_DECIMALS = 4

TIER_LABELS = {
    cfg.TIER_6: "TIER_6",
    cfg.TIER_12: "TIER_12",
    cfg.TIER_18: "TIER_18",
    cfg.TIER_30: "TIER_30",
}

TIERS_IN_ORDER = [
    cfg.TIER_6,
    cfg.TIER_12,
    cfg.TIER_18,
    cfg.TIER_30,
]

# =============================================================================
# Density parameters
# =============================================================================

BASE_LAMBDA = 3.0
DENSITY_SCALE = 6.0
TARGET_MONTHLY_TASKS_PER_SERVER = 10000.0
AMBIENT_LAMBDA = 0.20

ASSIGN_EXPANSION_FACTOR = 3.0

USER_CATEGORIES = [
    ("very_heavy", 2.6, 0.10),
    ("heavy", 1.7, 0.20),
    ("moderate", 1.0, 0.40),
    ("light", 0.55, 0.20),
    ("very_light", 0.25, 0.10),
]

EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# Device selection
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# =============================================================================
# Time functions
# =============================================================================

def time_of_day_multiplier(hour: int) -> float:
    if 0 <= hour < 6:
        return 0.15
    if 6 <= hour < 9:
        return 0.60
    if 9 <= hour < 17:
        return 1.00
    if 17 <= hour < 22:
        return 1.30
    return 0.40


def weekday_multiplier(is_weekend: bool) -> float:
    return 0.85 if is_weekend else 1.00


def priority_probabilities(hour: int, is_weekend: bool):
    if 22 <= hour or hour < 6:
        return (0.08, 0.22, 0.70)

    if is_weekend:
        return (0.12, 0.28, 0.60)

    if 6 <= hour < 9:
        return (0.12, 0.28, 0.60)

    if 9 <= hour < 17:
        return (0.15, 0.30, 0.55)

    return (0.12, 0.28, 0.60)


# =============================================================================
# GeoLife parsing
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
                    dt = datetime.strptime(
                        f"{date_str} {time_str}",
                        "%Y-%m-%d %H:%M:%S",
                    )
                except (ValueError, IndexError):
                    continue

                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue

                yield lat, lon, dt

    except OSError:
        return


def build_user_profile(
    user_dir: Path,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:

    traj_dir = user_dir / "Trajectory"

    profile = defaultdict(list)

    if not traj_dir.is_dir():
        return profile

    seen_this_bucket = defaultdict(set)

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


def read_all_user_profiles(data_dir: Path):

    profiles = {}

    user_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir()
    )

    for user_dir in user_dirs:
        user_id = user_dir.name
        profiles[user_id] = build_user_profile(user_dir)

    return profiles


# =============================================================================
# User categories
# =============================================================================

def classify_user_category(
    user_id: str,
    rng: random.Random,
) -> Tuple[str, float]:

    names = [c[0] for c in USER_CATEGORIES]
    mults = [c[1] for c in USER_CATEGORIES]
    weights = [c[2] for c in USER_CATEGORIES]

    local_rng = random.Random(f"{SEED}-{user_id}")

    idx = local_rng.choices(
        range(len(names)),
        weights=weights,
        k=1,
    )[0]

    return names[idx], mults[idx]


# =============================================================================
# Dates
# =============================================================================

def month_dates(
    start_date: date_cls,
    num_days: int,
) -> List[date_cls]:

    return [
        start_date + timedelta(days=i)
        for i in range(num_days)
    ]


# =============================================================================
# GPU Haversine
# =============================================================================

@torch.no_grad()
def haversine_matrix_gpu(
    points_latlon: torch.Tensor,
    server_latlon: torch.Tensor,
) -> torch.Tensor:

    """
    points_latlon:
        [N, 2] -> latitude, longitude

    server_latlon:
        [S, 2] -> latitude, longitude

    returns:
        [N, S] distances in km
    """

    points = torch.deg2rad(points_latlon)
    servers = torch.deg2rad(server_latlon)

    lat1 = points[:, 0:1]
    lon1 = points[:, 1:2]

    lat2 = servers[:, 0].unsqueeze(0)
    lon2 = servers[:, 1].unsqueeze(0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat1)
        * torch.cos(lat2)
        * torch.sin(dlon / 2.0) ** 2
    )

    a = torch.clamp(a, 0.0, 1.0)

    return 2.0 * EARTH_RADIUS_KM * torch.asin(
        torch.sqrt(a)
    )


# =============================================================================
# GPU server assignment
# =============================================================================

@torch.no_grad()
def build_gpu_assignment_cache(
    profiles,
    servers: List[dict],
    batch_size: int = 250_000,
):
    """
    Assign every unique rounded GPS coordinate to one server.

    The original code calculates Haversine distance repeatedly for every
    sampled event. This function calculates it once per unique coordinate.

    Returns:
        dict[(lat, lon)] = server_id
    """

    unique_coords = set()

    for profile in profiles.values():
        for candidates in profile.values():
            for lat, lon in candidates:
                unique_coords.add((lat, lon))

    if not unique_coords:
        return {}

    coords = list(unique_coords)

    point_tensor = torch.tensor(
        coords,
        dtype=torch.float32,
        device=DEVICE,
    )

    server_latlon = torch.tensor(
        [
            [s["lat"], s["lon"]]
            for s in servers
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    server_ids = [s["id"] for s in servers]

    coverage = torch.tensor(
        [s["coverage_km"] for s in servers],
        dtype=torch.float32,
        device=DEVICE,
    )

    assignment_cache = {}

    total = len(coords)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch = point_tensor[start:end]

        distances = haversine_matrix_gpu(
            batch,
            server_latlon,
        )

        # Global nearest server.
        nearest_dist, nearest_idx = distances.min(dim=1)

        # Strict coverage.
        strict_dist = distances.masked_fill(
            distances > coverage.unsqueeze(0),
            float("inf"),
        )

        strict_idx = strict_dist.argmin(dim=1)
        strict_min = strict_dist.min(dim=1).values

        # Expanded coverage.
        expanded_limit = coverage * ASSIGN_EXPANSION_FACTOR

        expanded_dist = distances.masked_fill(
            distances > expanded_limit.unsqueeze(0),
            float("inf"),
        )

        expanded_idx = expanded_dist.argmin(dim=1)
        expanded_min = expanded_dist.min(dim=1).values

        # Selection order:
        # strict -> expanded -> globally nearest
        chosen_idx = torch.where(
            torch.isfinite(strict_min),
            strict_idx,
            torch.where(
                torch.isfinite(expanded_min),
                expanded_idx,
                nearest_idx,
            ),
        )

        chosen_idx_cpu = chosen_idx.cpu().tolist()

        for local_i, server_index in enumerate(chosen_idx_cpu):
            assignment_cache[
                coords[start + local_i]
            ] = server_ids[server_index]

        del distances
        del strict_dist
        del expanded_dist

    return assignment_cache


# =============================================================================
# Resolve mobility events
# =============================================================================

def resolve_events(
    tier: int,
    profiles,
    dates: List[date_cls],
    py_rng: random.Random,
    assignment_cache,
):
    """
    Generates the same logical events as the original implementation.

    Location selection is still done with Python's RNG because each profile
    bucket contains a variable-length list. The expensive server-distance
    calculation is already eliminated by assignment_cache.
    """

    n_servers = cfg.SERVER_COUNT[tier]

    servers = cfg.SERVERS[:n_servers]

    events = []

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

                server_id = assignment_cache.get((lat, lon))

                if server_id is None:
                    # Should almost never happen because the cache was built
                    # from all profile coordinates.
                    continue

                hour = (slot * SLOT_MINUTES) // 60

                events.append(
                    (
                        user_id,
                        server_id,
                        d,
                        slot,
                        hour,
                        is_weekend,
                        wk_mult,
                    )
                )

    return events, [s["id"] for s in servers]


# =============================================================================
# GPU task generation
# =============================================================================

@torch.no_grad()
def generate_tier_tasks_gpu(
    tier: int,
    profiles,
    user_categories,
    dates: List[date_cls],
    py_rng: random.Random,
    assignment_batch_size: int,
):
    """
    GPU implementation of calibration + Poisson + multinomial aggregation.
    """

    n_servers = cfg.SERVER_COUNT[tier]

    servers = cfg.SERVERS[:n_servers]

    print(
        f"      Building GPU server-assignment cache "
        f"for {len(servers)} servers ..."
    )

    assignment_cache = build_gpu_assignment_cache(
        profiles=profiles,
        servers=servers,
        batch_size=assignment_batch_size,
    )

    print(
        f"      Cached {len(assignment_cache):,} unique GPS coordinates."
    )

    events, server_ids = resolve_events(
        tier=tier,
        profiles=profiles,
        dates=dates,
        py_rng=py_rng,
        assignment_cache=assignment_cache,
    )

    if not events:
        return {
            sid: {
                (d, slot): (0, 0, 0)
                for d in dates
                for slot in range(SLOTS_PER_DAY)
            }
            for sid in server_ids
        }

    server_to_index = {
        sid: i
        for i, sid in enumerate(server_ids)
    }

    # -------------------------------------------------------------------------
    # Convert events into compact tensors.
    # -------------------------------------------------------------------------

    n_events = len(events)

    server_indices = np.empty(n_events, dtype=np.int64)
    weights = np.empty(n_events, dtype=np.float32)
    hours = np.empty(n_events, dtype=np.int64)
    weekends = np.empty(n_events, dtype=np.bool_)

    # Each event gets a unique output slot index:
    # day_index * 96 + slot.
    output_indices = np.empty(n_events, dtype=np.int64)

    date_to_index = {
        d: i
        for i, d in enumerate(dates)
    }

    for i, (
        user_id,
        server_id,
        d,
        slot,
        hour,
        is_weekend,
        wk_mult,
    ) in enumerate(events):

        _, category_mult = user_categories[user_id]

        weights[i] = (
            category_mult
            * time_of_day_multiplier(hour)
            * wk_mult
        )

        server_indices[i] = server_to_index[server_id]
        hours[i] = hour
        weekends[i] = is_weekend

        output_indices[i] = (
            date_to_index[d] * SLOTS_PER_DAY
            + slot
        )

    server_indices_t = torch.from_numpy(
        server_indices
    ).to(DEVICE)

    weights_t = torch.from_numpy(
        weights
    ).to(DEVICE)

    output_indices_t = torch.from_numpy(
        output_indices
    ).to(DEVICE)

    hours_t = torch.from_numpy(
        hours
    ).to(DEVICE)

    weekends_t = torch.from_numpy(
        weekends
    ).to(DEVICE)

    # -------------------------------------------------------------------------
    # Calibration pass on GPU.
    # -------------------------------------------------------------------------

    server_weight_totals = torch.zeros(
        n_servers,
        dtype=torch.float32,
        device=DEVICE,
    )

    server_weight_totals.index_add_(
        0,
        server_indices_t,
        weights_t,
    )

    raw_expected = (
        BASE_LAMBDA
        * server_weight_totals
    )

    target = torch.full_like(
        raw_expected,
        TARGET_MONTHLY_TASKS_PER_SERVER,
    )

    server_calibration = torch.where(
        raw_expected > 0,
        target / raw_expected,
        torch.zeros_like(raw_expected),
    )

    # -------------------------------------------------------------------------
    # Poisson generation on GPU.
    # -------------------------------------------------------------------------

    lambda_values = (
        BASE_LAMBDA
        * weights_t
        * server_calibration[server_indices_t]
        * DENSITY_SCALE
    )

    n_tasks = torch.poisson(
        torch.clamp(lambda_values, min=0.0)
    ).to(torch.int64)

    # -------------------------------------------------------------------------
    # Priority probabilities.
    # -------------------------------------------------------------------------

    p_high = torch.empty(
        n_events,
        dtype=torch.float32,
        device=DEVICE,
    )

    p_mid = torch.empty_like(p_high)
    p_low = torch.empty_like(p_high)

    # Default evening.
    p_high.fill_(0.12)
    p_mid.fill_(0.28)
    p_low.fill_(0.60)

    # Night.
    night = (hours_t >= 22) | (hours_t < 6)

    p_high[night] = 0.08
    p_mid[night] = 0.22
    p_low[night] = 0.70

    # Weekend (only non-night hours).
    weekend_mask = (
        weekends_t
        & (~night)
    )

    p_high[weekend_mask] = 0.12
    p_mid[weekend_mask] = 0.28
    p_low[weekend_mask] = 0.60

    # Morning.
    morning = (
        (hours_t >= 6)
        & (hours_t < 9)
        & (~weekends_t)
    )

    p_high[morning] = 0.12
    p_mid[morning] = 0.28
    p_low[morning] = 0.60

    # Working hours.
    working = (
        (hours_t >= 9)
        & (hours_t < 17)
        & (~weekend_mask)
    )

    p_high[working] = 0.15
    p_mid[working] = 0.30
    p_low[working] = 0.55

    # -------------------------------------------------------------------------
    # Vectorized multinomial priority allocation.
    #
    # A multinomial(n, p) can be sampled efficiently using two binomial draws:
    #
    # high ~ Binomial(n, p_high)
    # mid  ~ Binomial(n-high, p_mid/(p_mid+p_low))
    # low  = remaining
    # -------------------------------------------------------------------------

    high = torch.zeros_like(n_tasks)

    # torch.distributions.Binomial supports vectorized GPU sampling.
    high_dist = torch.distributions.Binomial(
        total_count=n_tasks.to(torch.float32),
        probs=p_high,
    )

    high = high_dist.sample().to(torch.int64)

    remaining = n_tasks - high

    mid_probability = p_mid / torch.clamp(
        p_mid + p_low,
        min=1e-12,
    )

    mid_dist = torch.distributions.Binomial(
        total_count=remaining.to(torch.float32),
        probs=mid_probability,
    )

    mid = mid_dist.sample().to(torch.int64)

    low = remaining - mid

    # -------------------------------------------------------------------------
    # Aggregate directly on GPU.
    # Each output location has:
    #   server_index * total_slots + date_slot_index
    # -------------------------------------------------------------------------

    total_slots = len(dates) * SLOTS_PER_DAY

    flat_index = (
        server_indices_t * total_slots
        + output_indices_t
    )

    total_output_size = n_servers * total_slots

    high_out = torch.zeros(
        total_output_size,
        dtype=torch.int64,
        device=DEVICE,
    )

    mid_out = torch.zeros_like(high_out)
    low_out = torch.zeros_like(high_out)

    high_out.index_add_(
        0,
        flat_index,
        high,
    )

    mid_out.index_add_(
        0,
        flat_index,
        mid,
    )

    low_out.index_add_(
        0,
        flat_index,
        low,
    )

    # -------------------------------------------------------------------------
    # Ambient background.
    #
    # This is small compared with mobility-driven events, so we generate one
    # vector containing all server/date/slot combinations.
    # -------------------------------------------------------------------------

    ambient_slots = len(dates) * SLOTS_PER_DAY

    ambient_lambda = torch.empty(
        ambient_slots,
        dtype=torch.float32,
        device=DEVICE,
    )

    ambient_ph = torch.empty_like(ambient_lambda)
    ambient_pm = torch.empty_like(ambient_lambda)
    ambient_pl = torch.empty_like(ambient_lambda)

    for day_idx, d in enumerate(dates):

        is_weekend = d.weekday() >= 5

        wk_mult = weekday_multiplier(is_weekend)

        for slot in range(SLOTS_PER_DAY):

            idx = (
                day_idx * SLOTS_PER_DAY
                + slot
            )

            hour = (slot * SLOT_MINUTES) // 60

            ambient_lambda[idx] = (
                AMBIENT_LAMBDA
                * time_of_day_multiplier(hour)
                * wk_mult
            )

            ph, pm, pl = priority_probabilities(
                hour,
                is_weekend,
            )

            ambient_ph[idx] = ph
            ambient_pm[idx] = pm
            ambient_pl[idx] = pl

    ambient_lambda_all = ambient_lambda.repeat(n_servers)

    ambient_tasks = torch.poisson(
        torch.clamp(
            ambient_lambda_all,
            min=0.0,
        )
    ).to(torch.int64)

    ambient_ph_all = ambient_ph.repeat(n_servers)
    ambient_pm_all = ambient_pm.repeat(n_servers)
    ambient_pl_all = ambient_pl.repeat(n_servers)

    ambient_high = torch.distributions.Binomial(
        total_count=ambient_tasks.float(),
        probs=ambient_ph_all,
    ).sample().to(torch.int64)

    ambient_remaining = (
        ambient_tasks
        - ambient_high
    )

    ambient_mid_probability = (
        ambient_pm_all
        / torch.clamp(
            ambient_pm_all + ambient_pl_all,
            min=1e-12,
        )
    )

    ambient_mid = torch.distributions.Binomial(
        total_count=ambient_remaining.float(),
        probs=ambient_mid_probability,
    ).sample().to(torch.int64)

    ambient_low = (
        ambient_remaining
        - ambient_mid
    )

    high_out += ambient_high
    mid_out += ambient_mid
    low_out += ambient_low

    # -------------------------------------------------------------------------
    # Move only the final relatively-small result back to CPU.
    # -------------------------------------------------------------------------

    high_cpu = high_out.cpu().numpy()
    mid_cpu = mid_out.cpu().numpy()
    low_cpu = low_out.cpu().numpy()

    counts = {}

    for server_index, server_id in enumerate(server_ids):

        day_slot_map = {}

        base = server_index * total_slots

        for day_idx, d in enumerate(dates):

            for slot in range(SLOTS_PER_DAY):

                idx = (
                    base
                    + day_idx * SLOTS_PER_DAY
                    + slot
                )

                day_slot_map[
                    (d, slot)
                ] = (
                    int(high_cpu[idx]),
                    int(mid_cpu[idx]),
                    int(low_cpu[idx]),
                )

        counts[server_id] = day_slot_map

    # Free large GPU tensors before next tier.
    del server_indices_t
    del weights_t
    del output_indices_t
    del hours_t
    del weekends_t
    del server_weight_totals
    del server_calibration
    del lambda_values
    del n_tasks
    del high
    del mid
    del low
    del high_out
    del mid_out
    del low_out

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return counts


# =============================================================================
# CSV output
# =============================================================================

def write_server_csv(
    out_path: Path,
    dates: List[date_cls],
    slot_counts,
) -> None:

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "datetime",
                "high",
                "mid",
                "low",
            ]
        )

        for d in dates:

            for slot in range(SLOTS_PER_DAY):

                minute_of_day = (
                    slot * SLOT_MINUTES
                )

                dt = (
                    datetime(
                        d.year,
                        d.month,
                        d.day,
                    )
                    + timedelta(
                        minutes=minute_of_day
                    )
                )

                high, mid, low = slot_counts[
                    (d, slot)
                ]

                writer.writerow(
                    [
                        dt.strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        high,
                        mid,
                        low,
                    ]
                )


# =============================================================================
# Main
# =============================================================================

def main():

    global DENSITY_SCALE
    global TARGET_MONTHLY_TASKS_PER_SERVER
    global AMBIENT_LAMBDA

    parser = argparse.ArgumentParser(
        description=(
            "GPU-accelerated MEC task generation "
            "from GeoLife trajectories."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=cfg.TASK_PRODUCER_DIR / "Data",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.TASK_PRODUCER_DIR,
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
    )

    parser.add_argument(
        "--num-days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--density-scale",
        type=float,
        default=DENSITY_SCALE,
    )

    parser.add_argument(
        "--target-monthly-tasks",
        type=float,
        default=TARGET_MONTHLY_TASKS_PER_SERVER,
    )

    parser.add_argument(
        "--ambient-lambda",
        type=float,
        default=AMBIENT_LAMBDA,
    )

    parser.add_argument(
        "--assignment-batch-size",
        type=int,
        default=250_000,
        help=(
            "Number of unique GPS coordinates processed per GPU batch. "
            "Reduce this if GPU memory is insufficient."
        ),
    )

    args = parser.parse_args()

    DENSITY_SCALE = args.density_scale
    TARGET_MONTHLY_TASKS_PER_SERVER = (
        args.target_monthly_tasks
    )
    AMBIENT_LAMBDA = args.ambient_lambda

    # -------------------------------------------------------------------------
    # Random seeds
    # -------------------------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)

    py_rng = random.Random(SEED)

    # PyTorch RNG.
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    print("=" * 70)

    if DEVICE.type == "cuda":

        print(
            f"GPU acceleration enabled: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        print(
            "CUDA is not available. "
            "Running with PyTorch CPU fallback."
        )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------------------

    start_date = datetime.strptime(
        args.start_date,
        "%Y-%m-%d",
    ).date()

    dates = month_dates(
        start_date,
        args.num_days,
    )

    # -------------------------------------------------------------------------
    # Read trajectories
    # -------------------------------------------------------------------------

    print(
        f"[1/4] Reading trajectories from "
        f"{args.data_dir} ..."
    )

    profiles = read_all_user_profiles(
        args.data_dir
    )

    print(
        f"      Loaded profiles for "
        f"{len(profiles):,} users."
    )

    # -------------------------------------------------------------------------
    # User categories
    # -------------------------------------------------------------------------

    print(
        "[2/4] Classifying user activity categories ..."
    )

    user_categories = {
        uid: classify_user_category(
            uid,
            py_rng,
        )
        for uid in profiles
    }

    # -------------------------------------------------------------------------
    # Generate every tier
    # -------------------------------------------------------------------------

    for tier in TIERS_IN_ORDER:

        label = TIER_LABELS[tier]

        n_servers = cfg.SERVER_COUNT[tier]

        print()

        print(
            f"[3/4] Generating tasks for {label} "
            f"({n_servers} servers)"
        )

        print(
            f"      density_scale={DENSITY_SCALE}, "
            f"target/server={TARGET_MONTHLY_TASKS_PER_SERVER}, "
            f"ambient_lambda={AMBIENT_LAMBDA}"
        )

        server_task_counts = generate_tier_tasks_gpu(
            tier=tier,
            profiles=profiles,
            user_categories=user_categories,
            dates=dates,
            py_rng=py_rng,
            assignment_batch_size=args.assignment_batch_size,
        )

        # ---------------------------------------------------------------------
        # Write CSVs
        # ---------------------------------------------------------------------

        print(
            f"[4/4] Writing CSVs for {label} ..."
        )

        for i, srv in enumerate(
            cfg.SERVERS[:n_servers]
        ):

            out_path = (
                args.output_dir
                / label
                / f"tasks-{i}.csv"
            )

            write_server_csv(
                out_path,
                dates,
                server_task_counts[
                    srv["id"]
                ],
            )

            total = sum(
                sum(v)
                for v in server_task_counts[
                    srv["id"]
                ].values()
            )

            print(
                f"      server {i} "
                f"({srv['id']}): "
                f"{total:,} tasks/month"
            )

    print()
    print(
        f"Done. Output written under: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
