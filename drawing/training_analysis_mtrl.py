import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import config as cfg

# ===================== CONFIG =====================
N_SERVERS    = 5                      # number of servers
FIRST_SERVER = 1                      # servers numbered FIRST_SERVER .. FIRST_SERVER+N-1
DATA_DIR     = os.path.join(cfg.RESULT_DIR, "TIER_6", "LACT-Off")                   # folder containing the csv files
FILE_PATTERN = "[SENSITIVITY1_0_0]_LACT-Off_TOPO[0]training_log_server{n}.csv"
AGENTS       = [0, 1, 2]              # reward-0, reward-1, reward-2 ...
X_AXIS       = "index"                # "index" (row number) or "ex_time"
SMOOTH_WINDOW = 1                     # moving-average window (1 = no smoothing)
SAVE_PATH    = cfg.GRAPH_DIR / "training_summary_mtrl.png"

METRICS = {                           # title -> column prefix
    "Reward":              "reward",
    "Critic Loss":         "critic_loss",
    "Shared Critic Loss":  "shared_critic_loss",
}
# ==================================================


def minmax(x):
    """Min-max normalise a 1-D array to [0, 1] (NaN-safe)."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def load_server_series(prefix):
    """
    For every server: average the columns <prefix>-0, <prefix>-1, <prefix>-2 row-wise.
    Returns list of 1-D arrays (one per server) and the x-axis of the longest server.
    """
    series, xs = [], []
    for n in range(FIRST_SERVER, FIRST_SERVER + N_SERVERS):
        path = os.path.join(DATA_DIR, FILE_PATTERN.format(n=n))
        if not os.path.exists(path):
            print(f"[warn] missing file, skipped: {path}")
            continue
        df = pd.read_csv(path, sep=None, engine="python")   # auto-detects comma / tab
        df.columns = [c.strip() for c in df.columns]
        cols = [f"{prefix}-{a}" for a in AGENTS if f"{prefix}-{a}" in df.columns]
        if not cols:
            print(f"[warn] no '{prefix}' columns in {path}")
            continue

        y = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)   # avg over agents
        if SMOOTH_WINDOW > 1:
            y = y.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        series.append(y.to_numpy(dtype=float))
        xs.append(df["ex_time"].to_numpy(dtype=float) if X_AXIS == "ex_time"
                  else np.arange(len(df)))
    if not series:
        raise FileNotFoundError(f"No usable data found for '{prefix}'")
    return series, max(xs, key=len)


def aggregate(series):
    """Stack servers (NaN-padded to equal length) and compute mean, std, 95% CI across servers."""
    L = max(len(s) for s in series)
    data = np.full((len(series), L), np.nan)
    for i, s in enumerate(series):
        data[i, :len(s)] = s

    n    = np.sum(~np.isnan(data), axis=0)
    mean = np.nanmean(data, axis=0)
    std  = np.nanstd(data, axis=0, ddof=1) if len(series) > 1 else np.zeros(L)
    sem  = std / np.sqrt(np.maximum(n, 1))
    tcrit = np.array([stats.t.ppf(0.975, k - 1) if k > 1 else 0.0 for k in n])
    ci   = tcrit * sem
    return mean, std, ci


def plot_panel(ax, x, mean, std, ci, title, color):
    x = x[:len(mean)]
    ax.plot(x, mean, color=color, lw=1.8, label="Mean (across servers)")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20, label="± 1 Std Dev")
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.45, label="95% CI")
    ax.set_title(title)
    ax.set_xlabel("ex_time" if X_AXIS == "ex_time" else "Step / Row index")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main():
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 4.2 * len(METRICS)))
    colors = ["tab:blue", "tab:red", "tab:green"]

    for row, (name, prefix) in enumerate(METRICS.items()):
        series, x = load_server_series(prefix)

        # ---- raw: average agents -> per-server series -> mean/std/CI across servers
        mean, std, ci = aggregate(series)
        plot_panel(axes[row, 0], x, mean, std, ci, f"{name} (raw)", colors[row])
        axes[row, 0].set_ylabel(name)

        # ---- normalised: min-max each server's series to [0,1] first, then aggregate
        norm_series = [minmax(s) for s in series]
        nmean, nstd, nci = aggregate(norm_series)
        plot_panel(axes[row, 1], x, nmean, nstd, nci, f"{name} (normalized, min-max per server)", colors[row])
        axes[row, 1].set_ylabel(f"{name} (0-1)")

    fig.suptitle(f"Average over {N_SERVERS} servers (agents 0-2 averaged per server)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(SAVE_PATH, dpi=200)
    print(f"Saved figure to {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()