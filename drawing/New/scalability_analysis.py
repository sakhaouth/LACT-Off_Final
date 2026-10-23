"""SCALABILITY analysis: execution time and completion rate vs number of servers (sensitivity 0_0_0).

  average_execution_time = mean over iterations of ex_time
      PDMA: ex_time of the execution log; LACT_Off: ex_time averaged over the server logs.
  completion_rate = mean over iterations of sum(success)/sum(generation)  [%]

Graphs
  1. Execution vs Server number
  2. Completion rate vs Server number
Also writes SCALABILITY_summary.csv to the graph directory.

Usage:
  python scalability_analysis.py --tiers TIER_6 TIER_12 TIER_18 TIER_30
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import drawing.New.cfg as cfg
from drawing.New.common import Exp, base_parser, load_performance, load_training, n_servers, save_fig

ANALYSIS = "SCALABILITY"
EXPERIMENTS = ["PDMA", "LACT-Off"]


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--tiers", nargs="+", default=["TIER_6", "TIER_12", "TIER_18"])
    a = p.parse_args(argv)
    sens = (0, 0, 0)  # fixed by the experiment description

    rows = []
    for name in EXPERIMENTS:
        for tier in a.tiers:
            row = {"experiment": name, "tier": tier, "servers": n_servers(tier),
                   "avg_exec_time": float("nan"), "completion_rate": float("nan")}
            try:
                row["completion_rate"] = load_performance(Exp(name), tier, sens, a.align)["completion_rate"].mean()
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] {name} {tier} performance: {e}")
            try:
                row["avg_exec_time"] = load_training(Exp(name), tier, sens, a.align)["ex_time"].mean()
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] {name} {tier} training: {e}")
            rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("no data found")
        return
    out_dir = Path(cfg.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{ANALYSIS}_summary.csv", index=False)

    for col, ylabel, caption in [
        ("avg_exec_time", "Average execution time (s)", "Execution vs Server number"),
        ("completion_rate", "Completion rate (%)", "Completion rate vs Server number"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for name in EXPERIMENTS:
            s = summary[summary.experiment == name].dropna(subset=[col]).sort_values("servers")
            if not s.empty:
                ax.plot(s["servers"], s[col], marker="o", label=name)
        ax.set_xticks(sorted(summary["servers"].unique()))
        ax.set_xlabel("Number of server")
        ax.set_ylabel(ylabel)
        ax.set_title(caption)
        ax.grid(alpha=0.3)
        ax.legend()
        save_fig(fig, ANALYSIS, caption)


if __name__ == "__main__":
    main()
