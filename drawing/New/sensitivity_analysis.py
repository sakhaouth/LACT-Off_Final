"""SENSITIVITY analysis: completion rate vs CPU / memory / size sensitivity (10^0, 10^1, 10^2).

For each dimension only that exponent is varied; the other two stay at 0
(e.g. CPU sweep uses SENSITIVITY0_0_0, SENSITIVITY1_0_0, SENSITIVITY2_0_0).

<dim>_total_average_completion_rate = mean over servers of the per-server completion rate.
Mean and standard deviation are taken over the iterations of each run
(use --tail 0.2 to use only the last 20% of iterations, i.e. the converged part).

Graphs
  1. Completion Rate(%) VS Sensitivity  (3 sub graphs: cpu / memory / size; PDMA vs LACT_Off, mean +- std)
  2. Executed tasks VS Sensitivity      (same layout, average number of executed tasks +- std)
Also writes SENSITIVITY_summary.csv to the graph directory.

Usage:
  python sensitivity_analysis.py --tier TIER_6 --exponents 0 1 2
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import drawing.New.cfg as cfg
from drawing.New.common import Exp, base_parser, load_performance, save_fig

ANALYSIS = "SENSITIVITY"
DIMS = {"cpu": 0, "mem": 1, "size": 2}
DIM_TITLE = {"cpu": "CPU", "mem": "memory", "size": "Task size"}
EXPERIMENTS = ["PDMA", "LACT-Off"]


def sens_tuple(dim: str, exponent: int):
    t = [0, 0, 0]
    t[DIMS[dim]] = exponent
    return tuple(t)


def summarize(df: pd.DataFrame, tail: float):
    k = max(1, int(round(len(df) * tail)))
    df = df.iloc[-k:]
    return {
        "rate_mean": df["completion_rate_server_mean"].mean(),
        "rate_std": df["completion_rate_server_mean"].std(ddof=0),
        "tasks_mean": df["total_completed"].mean(),
        "tasks_std": df["total_completed"].std(ddof=0),
    }


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--exponents", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tail", type=float, default=1.0, help="fraction of final iterations to use (default 1.0 = all)")
    a = p.parse_args(argv)

    rows = []
    for name in EXPERIMENTS:
        for dim in DIMS:
            for e in a.exponents:
                try:
                    df = load_performance(Exp(name), a.tier, sens_tuple(dim, e), a.align)
                except (FileNotFoundError, ValueError) as err:
                    print(f"[warn] {name} {dim}={e}: {err}")
                    continue
                rows.append({"experiment": name, "dimension": dim, "exponent": e, **summarize(df, a.tail)})
    if not rows:
        print("no data found")
        return
    summary = pd.DataFrame(rows)
    out_dir = Path(cfg.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{ANALYSIS}_summary.csv", index=False)

    def figure(mean_col, std_col, ylabel, caption, sub_captions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, dim in zip(axes, DIMS):
            for name in EXPERIMENTS:
                s = summary[(summary.experiment == name) & (summary.dimension == dim)].sort_values("exponent")
                if s.empty:
                    continue
                ax.errorbar(s["exponent"], s[mean_col], yerr=s[std_col], marker="o", capsize=4, label=name)
            ax.set_xticks(a.exponents)
            ax.set_xlabel(f"{dim} exponent")
            ax.set_ylabel(ylabel)
            ax.set_title(sub_captions[dim])
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(caption)
        fig.tight_layout()
        save_fig(fig, ANALYSIS, caption)

    figure("rate_mean", "rate_std", "Completion rate (%)", "Completion Rate(%) VS Sensitivity",
           {d: f"Completion rate vs {DIM_TITLE[d]} sensitivity" for d in DIMS})
    figure("tasks_mean", "tasks_std", "Average number of executed task", "Executed task VS Sensitivity",
           {d: f"Executed task vs {DIM_TITLE[d]} sensitivity" for d in DIMS})


if __name__ == "__main__":
    main()
