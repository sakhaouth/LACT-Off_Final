"""PERFORMANCE and PERFORMANCE[HOMO VS HETERO] analysis.

Graphs
  1. total_completed_task vs iteration         (5 experiments)
  2. completion_rate vs iteration               (5 experiments)
  3. grouped bars: low/mid/high task time       (5 experiments)
  4. completion_rate vs iteration (HOMO vs HETERO, 4 experiments)
  5. grouped bars: low/mid/high task time (HOMO vs HETERO)

Usage:
  python performance_analysis.py                 # both analyses, TIER_6
  python performance_analysis.py --mode main --tier TIER_6 --smooth 5
  python performance_analysis.py --mode homo --logy   # log y-axis on the bar charts
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from drawing.New.common import CATEGORIES, Exp, base_parser, load_performance, save_fig, smooth

MAIN_EXPS = ["-PDMA", "PDMA", "-LACT_Off", "LACT-Off-", "LACT-Off"]
HOMO_EXPS = ["[HOMO]PDMA", "PDMA", "[HOMO]LACT-Off", "LACT-Off"]


def collect(labels, tier, sens, align):
    data = {}
    for lab in labels:
        exp = Exp.parse(lab)
        try:
            data[exp.label] = load_performance(exp, tier, sens, align)
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] skipping {lab}: {e}")
    return data


def line_plot(data, column, ylabel, analysis, caption, window):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in data.items():
        ax.plot(df.index, smooth(df[column], window), label=label, linewidth=1.5)
    ax.set_xlabel("Number of iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(caption)
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, analysis, caption)


def bar_plot(data, analysis, caption, logy=False):
    labels = list(data)
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, cat in enumerate(CATEGORIES):
        # average_<cat>_task_time: mean over iterations (and servers, workers) of <cat>_time
        vals = [data[l][f"{cat}_time"].mean() for l in labels]
        ax.bar(x + (i - 1) * w, vals, w, label=f"{cat.capitalize()} priority")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Execution time (Sec)")
    if logy:
        ax.set_yscale("log")
    ax.set_title(caption)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_fig(fig, analysis, caption)


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["main", "homo", "all"], default="all")
    p.add_argument("--logy", action="store_true", help="log-scale y-axis for the bar charts")
    a = p.parse_args(argv)

    if a.mode in ("main", "all"):
        name = "PERFORMANCE"
        data = collect(MAIN_EXPS, a.tier, tuple(a.sens), a.align)
        if data:
            line_plot(data, "total_completed", "Number of completed task", name,
                      "Number of task VS iteration", a.smooth)
            line_plot(data, "completion_rate", "Completion rate (%)", name,
                      "Completion rate VS iteration", a.smooth)
            bar_plot(data, name, "Execution time VS Experiment", a.logy)

    if a.mode in ("homo", "all"):
        name = "PERFORMANCE[HOMO VS HETERO]"
        data = collect(HOMO_EXPS, a.tier, tuple(a.sens), a.align)
        if data:
            line_plot(data, "completion_rate", "Completion rate (%)", name,
                      "Completion rate VS iteration (HOMO VS HETERO)", a.smooth)
            bar_plot(data, name, "Execution time VS Experiment (HOMO VS HETERO)", a.logy)


if __name__ == "__main__":
    main()
