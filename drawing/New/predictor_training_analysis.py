"""PREDICTOR_TRAINING analysis: LSTM vs Time-LLM (TIER_6 by default).

Reads every <TIER_n>train_predictor_log_bs{b}.csv found under <RESULT_DIR>/<TIER_n>/ (recursively),
averages each metric over the base stations b, then min-max normalizes the losses.

Iteration counts differ between files (and between the LSTM / LLM columns of one file):
  --align common (default): only iterations present in ALL files are averaged
  --align union           : every iteration is averaged over the files that have it
NaNs (e.g. one model stopped earlier by early stopping) are dropped per column.

Graph (one figure, 3 sub graphs): training loss, validation loss, execution time vs iteration.

Usage:
  python predictor_training_analysis.py --tier TIER_6
"""
from __future__ import annotations

from pathlib import Path
from glob import escape
import matplotlib.pyplot as plt
import pandas as pd

import drawing.New.cfg as cfg
from drawing.New.common import align_mean, base_parser, minmax, read_csv, save_fig, smooth, tier_name

ANALYSIS = "PREDICTOR_TRAINING"
MODELS = {"LSTM": "lstm", "Time-LLM": "llm"}
METRICS = {  # key -> (csv suffix, normalize?, y label, sub caption)
    "train": ("train_loss", False, "Training loss(Log Scale)", "Training Loss vs Number of iteration"),
    "val": ("test_loss", False, "Validation loss(Log Scale)", "Validation Loss vs Number of iteration"),
    "time": ("training_time", False, "Average execution time(Log Scale)", "Execution time vs Number of iteration"),
}


def load_predictor(tier: str, align: str) -> dict[str, pd.Series]:
    print(f"TIER: {tier}")

    pattern = f"{escape(f'[{tier}]')}train_predictor_log_bs*.csv"
    files = sorted(Path(cfg.RESULT_DIR).rglob(pattern))

    if not files:
        raise FileNotFoundError(
            f"no '[{tier}]train_predictor_log_bs*.csv' "
            f"under {cfg.RESULT_DIR}"
        )
    print(files)
    dfs = [read_csv(f) for f in files]
    print(f"Found {len(files)} predictor logs")

    out = {}

    for prefix in MODELS.values():
        for suffix, *_ in METRICS.values():
            col = f"{prefix}_{suffix}"

            # Keep non-empty columns from files containing this metric.
            print(f"COLS is {col}")
            frames = [
                df[[col]].dropna()
                for df in dfs
                if col in df.columns
            ]
            frames = [frame for frame in frames if not frame.empty]

            if frames:
                # Use the union of iteration indices.
                # Each iteration is averaged over available values only.
                out[col] = align_mean(frames, "union")[col]

    return out


def main(argv=None):
    p = base_parser(__doc__.splitlines()[0])
    a = p.parse_args(argv)
    tier = tier_name(a.tier)
    try:
        series = load_predictor(tier, a.align)

    except FileNotFoundError as e:
        print(f"[warn] {e}")
        return

    caption = "Predictor Test loss, validation loss and Execution time"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (suffix, normalize, ylabel, sub) in zip(axes, METRICS.values()):
        for label, prefix in MODELS.items():
            s = series.get(f"{prefix}_{suffix}")
            if s is None:
                continue
            s = smooth(s, a.smooth)
            ax.set_yscale("log")
            ax.plot(s.index, minmax(s) if normalize else s, label=label, linewidth=1.5)
        ax.set_xlabel("Number of iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(sub)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    save_fig(fig, ANALYSIS, caption)


if __name__ == "__main__":
    main()
