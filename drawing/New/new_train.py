"""TRAINING analysis: min-max normalized reward and critic loss, PDMA vs LACT_Off,
with mean, ±1 standard deviation and 95% confidence interval across servers.

Graphs
  1. Normalized reward VS iteration        (PDMA, LACT_Off)
  2. Normalized loss VS iteration          (LACT_Off critic, LACT_Off shared critic [, PDMA critic])

Method
  - Each server's curve is smoothed (moving average, --smooth).
  - The mean curve over servers is min-max normalized (same as the original script); the same
    affine transform (x - min) / (max - min) is then applied to every server curve, so the
    plotted mean is identical to the old plot and the bands are on the same scale.
  - Per iteration: std = sample std over servers (ddof=1),
                   95% CI of the mean = t_{0.975, k-1} * std / sqrt(k), k = number of servers.

Usage:
  python training_analysis.py --tier TIER_6 --smooth 5
  python training_analysis.py --tier TIER_6 --smooth 5 --with-pdma-loss
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from drawing.New.common import (
    Exp, base_parser, exp_dir, locate, n_servers, read_csv, save_fig,
    sens_tag, smooth, tier_name, tier_to_idx,
)
import re

ANALYSIS = "TRAINING"


# --------------------------------------------------------------------------- #
# Per-server loader (iteration x server DataFrame for each metric)
# --------------------------------------------------------------------------- #
def load_per_server(exp: Exp, tier, sens, align: str = "common") -> dict[str, pd.DataFrame]:
    tier, n = tier_name(tier), n_servers(tier)
    d, tag, idx = exp_dir(tier, exp.name), sens_tag(sens), tier_to_idx(n)
    series: dict[str, dict[int, pd.Series]] = {}

    if exp.family == "pdma":
        df = read_csv(locate(d, f"[{tag}]_{exp.name}_TOPO[{idx}]_execution_log.csv", exp.homo)[0])
        for b in range(n):
            for key in ("reward", "critic_loss"):
                col = f"server_{b}_{key}"
                if col in df.columns:
                    series.setdefault(key, {})[b] = df[col].astype(float)
    else:
        for b in range(n):
            hits = locate(d, f"[{tag}]_{exp.name}_TOPO[{idx}]training_log_server{b}.csv",
                          exp.homo, required=False)
            if not hits:
                continue
            df = read_csv(hits[0])

            def cols(prefix):
                return [c for c in df.columns if re.fullmatch(rf"{prefix}-\d+", c)]

            series.setdefault("reward", {})[b] = df[cols("reward")].astype(float).sum(axis=1, min_count=1)
            series.setdefault("critic_loss", {})[b] = df[cols("critic_loss")].astype(float).sum(axis=1, min_count=1)
            sc = [c for c in df.columns if "shared_critic_loss" in str(c).lower()]
            if sc:
                series.setdefault("shared_critic_loss", {})[b] = df[sc].astype(float).mean(axis=1)

    if not series:
        raise FileNotFoundError(f"No per-server training logs found for {exp.name} in {d}")

    out = {}
    for key, per_b in series.items():
        if align == "common":
            common = None
            for s in per_b.values():
                common = s.index if common is None else common.intersection(s.index)
            per_b = {b: s.loc[common] for b, s in per_b.items()}
        out[key] = pd.DataFrame(per_b).sort_index()
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def band_stats(df: pd.DataFrame, window: int) -> pd.DataFrame | None:
    """df: iteration x server. Returns mean, std, ci (half-width) on the normalized scale."""
    if df is None or df.empty or not df.notna().any().any():
        return None
    df = df.apply(lambda s: smooth(s, window))
    mean = df.mean(axis=1)
    lo, hi = mean.min(), mean.max()
    scale = hi - lo if np.isfinite(lo) and (hi - lo) > 1e-12 else 1.0
    df = (df - lo) / scale

    k = df.shape[1]
    mean = df.mean(axis=1)
    if k > 1:
        std = df.std(axis=1, ddof=1)
        ci = stats.t.ppf(0.975, k - 1) * std / np.sqrt(k)
    else:
        std = pd.Series(0.0, index=df.index)
        ci = std.copy()
    return pd.DataFrame({"mean": mean, "std": std, "ci": ci})


def plot_curve(ax, st: pd.DataFrame, label: str):
    (line,) = ax.plot(st.index, st["mean"], label=label, linewidth=1.5)
    c = line.get_color()
    ax.fill_between(st.index, st["mean"] - st["std"], st["mean"] + st["std"],
                    color=c, alpha=0.12, linewidth=0)
    ax.fill_between(st.index, st["mean"] - st["ci"], st["mean"] + st["ci"],
                    color=c, alpha=0.35, linewidth=0)


def finish(ax, ylabel, caption, k_note):
    ax.set_xlabel("Number of iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(caption)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="gray", alpha=0.12, label="±1 std"),
                Patch(facecolor="gray", alpha=0.35, label=f"95% CI of mean ({k_note})")]
    ax.legend(handles=handles)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--with-pdma-loss", action="store_true", help="also plot PDMA critic loss")
    a = p.parse_args(argv)
    sens = tuple(a.sens)
    k_note = f"over {a.tier} servers"

    data = {}
    for name in ("PDMA", "LACT-Off"):
        try:
            data[name] = load_per_server(Exp(name), a.tier, sens, a.align)
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] skipping {name}: {e}")

    # 1st graph: normalized reward
    if data:
        caption = "Normalized reward VS iteration"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, metrics in data.items():
            st = band_stats(metrics.get("reward"), a.smooth)
            if st is not None:
                plot_curve(ax, st, name)
        finish(ax, "Normalized average reward", caption, k_note)
        save_fig(fig, ANALYSIS, caption)

    # 2nd graph: normalized critic losses
    curves = []
    if a.with_pdma_loss and "PDMA" in data:
        curves.append(("PDMA critic loss", data["PDMA"].get("critic_loss")))
    if "LACT-Off" in data:
        curves.append(("LACT_Off critic loss", data["LACT-Off"].get("critic_loss")))
        curves.append(("LACT_Off shared critic loss", data["LACT-Off"].get("shared_critic_loss")))
    if curves:
        caption = "Normalized loss VS iteration"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, df in curves:
            st = band_stats(df, a.smooth)
            if st is not None:
                plot_curve(ax, st, label)
        finish(ax, "Normalized Critic loss", caption, k_note)
        save_fig(fig, ANALYSIS, caption)


if __name__ == "__main__":
    main()