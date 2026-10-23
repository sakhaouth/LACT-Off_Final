"""COMPLETION analysis (number and rate of completed tasks, with ±1 standard deviation).

Graphs
  1. total_completed_task vs iteration                          (5 experiments)
  2. completion_rate vs iteration                               (5 experiments)
  3. completion_rate vs iteration (HOMO vs HETERO)              (4 experiments)

Standard deviation (--band-source)
  window : spread over iterations. Runs are averaged first, then the rolling mean
           +- rolling std over `--window` iterations is drawn.
           Works for every experiment (also single-run ones such as PDMA).
  runs   : spread across independent runs (e.g. LACT_Off workers), sample std (ddof=1).
           Experiments with <2 runs get no band. `--smooth` is applied to each run first.

Usage:
  python completion_analysis.py
  python completion_analysis.py --mode main --tier TIER_6 --band-source runs --smooth 5
  python completion_analysis.py --mode homo --window 30
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from drawing.New.common import (
    Exp, base_parser, exp_dir, locate, n_servers, perf_frame,
    read_csv, save_fig, sens_tag, smooth, tier_name, tier_to_idx,
)

MAIN_EXPS = ["-PDMA", "PDMA", "-LACT_Off", "LACT-Off-", "LACT-Off"]
HOMO_EXPS = ["[HOMO]PDMA", "PDMA", "[HOMO]LACT-Off", "LACT-Off"]


# --------------------------------------------------------------------------- #
# Loading (keeps individual runs instead of averaging them)
# --------------------------------------------------------------------------- #
def align_runs(frames, mode="common"):
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index) if mode == "common" else idx.union(f.index)
    return [f.reindex(idx) if mode == "union" else f.loc[idx] for f in frames]


def load_runs(exp: Exp, tier, sens, align) -> list[pd.DataFrame]:
    """List of per-iteration performance tables, one per run (PDMA: 1, LACT_Off: one per worker)."""
    tier, n = tier_name(tier), n_servers(tier)
    d, tag, idx = exp_dir(tier, exp.name), sens_tag(sens), tier_to_idx(n)
    if exp.family == "pdma":
        paths = locate(d, f"[{tag}]_{exp.name}_TOPO[{idx}]_training_log.csv", exp.homo)[:1]
    else:
        paths = locate(d, f"[{tag}]_{exp.name}_TOPO[{idx}]_performance_parameter_worker*.csv", exp.homo)
    return align_runs([perf_frame(read_csv(p), n) for p in paths], align)


def collect(labels, tier, sens, align):
    data = {}
    for lab in labels:
        exp = Exp.parse(lab)
        try:
            data[exp.label] = load_runs(exp, tier, sens, align)
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] skipping {lab}: {e}")
    return data


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def line_stats(runs, column, source, window, smooth_w) -> pd.DataFrame:
    """DataFrame indexed by iteration with columns mean and std."""
    if source == "runs":
        mat = pd.concat([smooth(r[column], smooth_w) for r in runs], axis=1)
        mean, std = mat.mean(axis=1), mat.std(axis=1, ddof=1)
    else:  # window: average the runs first, then rolling statistics over iterations
        s = pd.concat([r[column] for r in runs], axis=1).mean(axis=1)
        roll = s.rolling(window, min_periods=2)
        mean, std = roll.mean(), roll.std(ddof=1)
    return pd.DataFrame({"mean": mean, "std": std})


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def line_plot(data, column, ylabel, analysis, caption, a):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, runs in data.items():
        if a.band_source == "runs" and len(runs) < 2:
            print(f"[warn] {label}: only 1 run, no std band in 'runs' mode")
        st = line_stats(runs, column, a.band_source, a.window, a.smooth)
        (line,) = ax.plot(st.index, st["mean"], label=label, linewidth=1.5)
        ax.fill_between(st.index, st["mean"] - st["std"], st["mean"] + st["std"],
                        color=line.get_color(), alpha=0.2, linewidth=0)
    ax.set_xlabel("Number of iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{caption}\n(mean ± 1 std)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, analysis, caption)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["main", "homo", "all"], default="all")
    p.add_argument("--band-source", choices=["window", "runs"], default="window",
                   help="std over iterations (window) or across runs/workers (runs)")
    p.add_argument("--window", type=int, default=20,
                   help="rolling window (iterations) for 'window' mode (default 20)")
    a = p.parse_args(argv)
    sens = tuple(a.sens)

    if a.mode in ("main", "all"):
        name = "PERFORMANCE"
        data = collect(MAIN_EXPS, a.tier, sens, a.align)
        if data:
            line_plot(data, "total_completed", "Number of completed task", name,
                      "Number of task VS iteration", a)
            line_plot(data, "completion_rate", "Completion rate (%)", name,
                      "Completion rate VS iteration", a)

    if a.mode in ("homo", "all"):
        name = "PERFORMANCE[HOMO VS HETERO]"
        data = collect(HOMO_EXPS, a.tier, sens, a.align)
        if data:
            line_plot(data, "completion_rate", "Completion rate (%)", name,
                      "Completion rate VS iteration (HOMO VS HETERO)", a)


if __name__ == "__main__":
    main()