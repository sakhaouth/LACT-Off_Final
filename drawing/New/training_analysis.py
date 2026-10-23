"""TRAINING analysis: min-max normalized reward and critic loss, PDMA vs LACT_Off.

Graphs
  1. Normalized reward VS iteration        (PDMA, LACT_Off)
  2. Normalized loss VS iteration          (PDMA critic, LACT_Off critic, LACT_Off shared critic)

Note: LACT_Off per-server values are the SUM over tasks d (as in the description). Because the
curves are min-max normalized, summing or averaging over tasks gives identical plots.

Usage:
  python training_analysis.py --tier TIER_6 --smooth 5
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from drawing.New.common import Exp, base_parser, load_training, minmax, save_fig, smooth

ANALYSIS = "TRAINING"


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    a = p.parse_args(argv)
    sens = tuple(a.sens)
    # print(sens)
    data = {}
    for name in ("PDMA", "LACT-Off"):
        try:
            data[name] = load_training(Exp(name), a.tier, sens, a.align)
            # print(data[name])
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] skipping {name}: {e}")

    def norm(series):
        return minmax(smooth(series, a.smooth))

    # 1st graph: normalized reward
    if data:
        caption = "Normalized reward VS iteration"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, df in data.items():
            ax.plot(df.index, norm(df["reward"]), label=name, linewidth=1.5)
        ax.set_xlabel("Number of iteration")
        ax.set_ylabel("Normalized average reward")
        ax.set_title(caption)
        ax.grid(alpha=0.3)
        ax.legend()
        save_fig(fig, ANALYSIS, caption)

    # 2nd graph: normalized critic losses
    curves = []
    # if "PDMA" in data:
    #     curves.append(("PDMA critic loss", data["PDMA"]["critic_loss"]))
    if "LACT-Off" in data:
        # print(data["LACT-Off"])
        curves.append(("LACT_Off critic loss", data["LACT-Off"]["critic_loss"]))
        curves.append(("LACT_Off shared critic loss", data["LACT-Off"]["shared_critic_loss"]))
    if curves:
        caption = "Normalized loss VS iteration"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, s in curves:
            if s.notna().any():
                ax.plot(s.index, norm(s), label=label, linewidth=1.5)
        ax.set_xlabel("Number of iteration")
        ax.set_ylabel("Normalized Critic loss")
        ax.set_title(caption)
        ax.grid(alpha=0.3)
        ax.legend()
        save_fig(fig, ANALYSIS, caption)


if __name__ == "__main__":
    main()
