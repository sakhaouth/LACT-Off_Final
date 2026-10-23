"""SENSITIVITY analysis: completion rate vs CPU / memory / size sensitivity (10^0, 10^1, 10^2).

For each dimension only that exponent is varied; the other two stay at 0
(e.g. CPU sweep uses SENSITIVITY0_0_0, SENSITIVITY1_0_0, SENSITIVITY2_0_0).

<dim>_total_average_completion_rate = mean over servers of the per-server completion rate.
Mean, standard deviation and 95% confidence interval are taken over the iterations of each run
(use --tail 0.2 to use only the last 20% of iterations, i.e. the converged part).

  std = sample standard deviation over iterations (ddof=1)
  CI  = t_{0.975, n-1} * std / sqrt(n)   (95% CI of the mean, n = number of iterations used)

Graphs
  1. Completion Rate(%) VS Sensitivity  (3 sub graphs: cpu / memory / size; PDMA vs LACT_Off,
     mean, +-1 std (thin bars) and 95% CI (thick bars))
  2. Executed tasks VS Sensitivity      (same layout, average number of executed tasks)
Also writes SENSITIVITY_summary.csv to the graph directory.

Usage:
  python sensitivity_analysis.py --tier TIER_6 --exponents 0 1 2 --tail 0.2
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import drawing.New.cfg as cfg
from drawing.New.common import Exp, base_parser, load_performance, save_fig

ANALYSIS = "SENSITIVITY"
DIMS = {"cpu": 0, "mem": 1, "size": 2}
DIM_TITLE = {"cpu": "CPU", "mem": "memory", "size": "Task size"}
EXPERIMENTS = ["PDMA", "LACT-Off"]
DODGE = 0.04  # horizontal offset between methods so error bars do not overlap


def sens_tuple(dim: str, exponent: int):
    t = [0, 0, 0]
    t[DIMS[dim]] = exponent
    return tuple(t)


def mean_std_ci(x: pd.Series):
    x = x.dropna().astype(float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = x.mean()
    if n < 2:
        return mean, 0.0, 0.0, n
    std = x.std(ddof=1)
    ci = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
    return mean, std, ci, n


def summarize(df: pd.DataFrame, tail: float):
    k = max(1, int(round(len(df) * tail)))
    df = df.iloc[-k:]
    rm, rs, rc, n = mean_std_ci(df["completion_rate_server_mean"])
    tm, ts, tc, _ = mean_std_ci(df["total_completed"])
    return {
        "n_iter": n,
        "rate_mean": rm, "rate_std": rs, "rate_ci": rc,
        "tasks_mean": tm, "tasks_std": ts, "tasks_ci": tc,
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

    def figure(prefix, ylabel, caption, sub_captions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        offsets = np.linspace(-DODGE, DODGE, len(EXPERIMENTS)) if len(EXPERIMENTS) > 1 else [0.0]
        for ax, dim in zip(axes, DIMS):
            for name, off in zip(EXPERIMENTS, offsets):
                s = summary[(summary.experiment == name) & (summary.dimension == dim)].sort_values("exponent")
                if s.empty:
                    continue
                x = s["exponent"].to_numpy() + off
                # +-1 std: thin bars, wide caps
                cont = ax.errorbar(x, s[f"{prefix}_mean"], yerr=s[f"{prefix}_std"], marker="o",
                                   capsize=6, linewidth=1.0, elinewidth=1.0, label=name)
                color = cont[0].get_color()
                # 95% CI of the mean: thick bars, small caps, drawn on top
                ax.errorbar(x, s[f"{prefix}_mean"], yerr=s[f"{prefix}_ci"], fmt="none",
                            ecolor=color, elinewidth=3.5, capsize=3, capthick=1.5, zorder=3)
            ax.set_xticks(a.exponents)
            ax.set_xlabel(f"{dim} exponent")
            ax.set_ylabel(ylabel)
            ax.set_title(sub_captions[dim])
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            handles += [
                Line2D([0], [0], color="gray", lw=1.0, label="±1 std"),
                Line2D([0], [0], color="gray", lw=3.5, label="95% CI of mean"),
            ]
            ax.legend(handles=handles)
        fig.suptitle(caption)
        fig.tight_layout()
        save_fig(fig, ANALYSIS, caption)

    figure("rate", "Completion rate (%)", "Completion Rate(%) VS Sensitivity",
           {d: f"Completion rate vs {DIM_TITLE[d]} sensitivity" for d in DIMS})
    figure("tasks", "Average number of executed task", "Executed task VS Sensitivity",
           {d: f"Executed task vs {DIM_TITLE[d]} sensitivity" for d in DIMS})


if __name__ == "__main__":
    main()