"""
plot_results.py
================
Generates comparison graphs (success/failure counts, success/failure rate,
smoothed rates, and per-mode success-vs-failure) across the 4 training
modes that are actually being logged:

    label_1 = MINUS_PDMA      -> "_training_log.csv"             (server_i_total_success / server_i_total_violation columns)
    label_2 = MINUS_LACT_Off  -> "_performance_parameter_worker0.csv" (success_i / time_out_i columns)
    label_3 = PDMA            -> "_training_log.csv"             (server_i_total_success / server_i_total_violation columns)
    label_4 = LACT_Off_MINUS  -> "_performance_parameter_worker0.csv" (success_i / time_out_i columns)

FIXES applied to the original draft:
  - `NUM_SERVERS` is already an int (`cfg.NUM_SERVERS`); the completion-time
    column builders were doing `range(len(NUM_SERVERS))`, which crashes
    (`int has no len()`). Changed to `range(NUM_SERVERS)`.
  - The two log formats have DIFFERENT success/failure column names
    (`server_i_total_success/violation` for the "_training_log.csv" files
    vs `success_i/time_out_i` for the "_performance_parameter_worker0.csv"
    files). The original code mixed these up (e.g. computed
    `pdma_success_rate` from an undefined `pdma_totla_success`, and applied
    `lact_off_success_cols` to `minus_pdma_df`). Every aggregate below now
    reads its OWN dataframe with its OWN matching column set.
  - Removed the undefined `maddpg_df`, `mtrl_df`, `lact_off__df`,
    `pdma_totla_success/failure` references (label_5 / LACT_Off_LOG stays
    commented out since its CSV read was never enabled -- uncomment
    `loct_off_df = pd.read_csv(LACT_Off_LOG)` and extend the plot lists
    below if/when that 5th run is available).
  - Success/failure RATE is now computed uniformly as
    success / (success + failure + eps) for all 4 modes (the original mixed
    this formula with a `total_generation`-based one that used a
    differently-indexed dataframe -- not safe to divide across dataframes
    of possibly different lengths).
  - Every figure now uses an explicit, consistent color per label (so the
    same mode is always the same color across all 8 figures), and legends
    always use the real `label_*` variables instead of hardcoded strings.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

# =====================================================
# Configuration
# =====================================================
NUM_SERVERS = cfg.NUM_SERVERS

label_1 = cfg.MODE_NAME[cfg.MINUS_PDMA]
label_2 = cfg.MODE_NAME[cfg.MINUS_LACT_Off]
label_3 = cfg.MODE_NAME[cfg.PDMA]
label_4 = cfg.MODE_NAME[cfg.LACT_Off_MINUS]
label_5 = cfg.MODE_NAME[cfg.LACT_Off]

MINUS_PDMA_LOG = cfg.RESULT_DIR / f"{label_1}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
MINUS_LACT_Off_LOG = cfg.RESULT_DIR / f"{label_2}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
PDMA_LOG = cfg.RESULT_DIR / f"{label_3}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
LACT_Off_MINUS_LOG = cfg.RESULT_DIR / f"{label_4}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
LACT_Off_LOG = cfg.RESULT_DIR / f"{label_5}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"

OUTPUT_DIR = cfg.GRAPH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One fixed color per mode, reused across every figure below.
COLOR = {
    label_1: "tab:blue",
    label_2: "tab:orange",
    label_3: "tab:green",
    label_4: "tab:red",
    label_5: "tab:purple",
}

EPS = 1e-8

# =====================================================
# Read CSV files
# =====================================================
minus_pdma_df = pd.read_csv(MINUS_PDMA_LOG)
minus_lact_off_df = pd.read_csv(MINUS_LACT_Off_LOG)
pdma_df = pd.read_csv(PDMA_LOG)
lact_off_minus_df = pd.read_csv(LACT_Off_MINUS_LOG)
lact_off_df = pd.read_csv(LACT_Off_LOG)
# 5th run not enabled by default -- uncomment when the CSV exists:
# lact_off_df = pd.read_csv(LACT_Off_LOG)

# =====================================================
# Column Names
# =====================================================
# "_training_log.csv" format (label_1 / label_3): per-server totals.
lact_off_success_cols = [f"success_{i}" for i in range(NUM_SERVERS)]
lact_off_failure_cols = [f"time_out_{i}" for i in range(NUM_SERVERS)]
pdma_success_cols = [f"server_{i}_total_success" for i in range(NUM_SERVERS)]
pdma_failor_cols = [f"server_{i}_total_violation" for i in range(NUM_SERVERS)]
# pdma_success_cols = lact_off_success_cols
# pdma_failor_cols = lact_off_failure_cols

# "_performance_parameter_worker0.csv" format (label_2 / label_4): per-server counters.

generation_cols = [f"generation_{i}" for i in range(NUM_SERVERS)]

# Per-priority completion-time columns.
# "_training_log.csv" format (label_1 / label_3): "low-{b}" / "mid-{b}" / "high-{b}"
pdma_low_cols = [f"low-{b}" for b in range(NUM_SERVERS)]
pdma_mid_cols = [f"mid-{b}" for b in range(NUM_SERVERS)]
pdma_high_cols = [f"high-{b}" for b in range(NUM_SERVERS)]
pdma_completion_time_cols = pdma_low_cols + pdma_mid_cols + pdma_high_cols

# "_performance_parameter_worker0.csv" format (label_2 / label_4): "low_time_{b}" / "mid_time_{b}" / "high_time_{b}"
lact_off_low_cols = [f"low_time_{b}" for b in range(NUM_SERVERS)]
lact_off_mid_cols = [f"mid_time_{b}" for b in range(NUM_SERVERS)]
lact_off_high_cols = [f"high_time_{b}" for b in range(NUM_SERVERS)]
lact_off_completion_time_cols = lact_off_low_cols + lact_off_mid_cols + lact_off_high_cols

# pdma_low_cols = lact_off_low_cols
# pdma_mid_cols = lact_off_mid_cols
# pdma_high_cols = lact_off_high_cols
# pdma_completion_time_cols = lact_off_completion_time_cols
# =====================================================
# Aggregates: total success / total failure per mode
# =====================================================
succ_1 = minus_pdma_df[pdma_success_cols].sum(axis=1).to_numpy()
fail_1 = minus_pdma_df[pdma_failor_cols].sum(axis=1).to_numpy()

succ_2 = minus_lact_off_df[lact_off_success_cols].sum(axis=1).to_numpy()
fail_2 = minus_lact_off_df[lact_off_failure_cols].sum(axis=1).to_numpy()

succ_3 = pdma_df[pdma_success_cols].sum(axis=1).to_numpy()
fail_3 = pdma_df[pdma_failor_cols].sum(axis=1).to_numpy()

succ_4 = lact_off_minus_df[lact_off_success_cols].sum(axis=1).to_numpy()
fail_4 = lact_off_minus_df[lact_off_failure_cols].sum(axis=1).to_numpy()

succ_5 = lact_off_df[lact_off_success_cols].sum(axis=1).to_numpy()
fail_5 = lact_off_df[lact_off_failure_cols].sum(axis=1).to_numpy()

# =====================================================
# Rates: success / (success + failure), consistently for every mode
# =====================================================
rate_1 = succ_1 / (succ_1 + fail_1 + EPS)
rate_2 = succ_2 / (succ_2 + fail_2 + EPS)
rate_3 = succ_3 / (succ_3 + fail_3 + EPS)
rate_4 = succ_4 / (succ_4 + fail_4 + EPS)
rate_5 = succ_5 / (succ_5 + fail_5 + EPS)

fail_rate_1 = fail_1 / (succ_1 + fail_1 + EPS)
fail_rate_2 = fail_2 / (succ_2 + fail_2 + EPS)
fail_rate_3 = fail_3 / (succ_3 + fail_3 + EPS)
fail_rate_4 = fail_4 / (succ_4 + fail_4 + EPS)
fail_rate_5 = fail_5 / (succ_5 + fail_5 + EPS)
# =====================================================
# Average completion time per priority (low/mid/high), averaged across
# servers, per training iteration -- one series per mode per priority.
# =====================================================
low_1 = minus_pdma_df[pdma_low_cols].mean(axis=1).to_numpy()
mid_1 = minus_pdma_df[pdma_mid_cols].mean(axis=1).to_numpy()
high_1 = minus_pdma_df[pdma_high_cols].mean(axis=1).to_numpy()

low_2 = minus_lact_off_df[lact_off_low_cols].mean(axis=1).to_numpy()
mid_2 = minus_lact_off_df[lact_off_mid_cols].mean(axis=1).to_numpy()
high_2 = minus_lact_off_df[lact_off_high_cols].mean(axis=1).to_numpy()

low_3 = pdma_df[pdma_low_cols].mean(axis=1).to_numpy()
mid_3 = pdma_df[pdma_mid_cols].mean(axis=1).to_numpy()
high_3 = pdma_df[pdma_high_cols].mean(axis=1).to_numpy()

low_4 = lact_off_minus_df[lact_off_low_cols].mean(axis=1).to_numpy()
mid_4 = lact_off_minus_df[lact_off_mid_cols].mean(axis=1).to_numpy()
high_4 = lact_off_minus_df[lact_off_high_cols].mean(axis=1).to_numpy()

low_5 = lact_off_df[lact_off_low_cols].mean(axis=1).to_numpy()
mid_5 = lact_off_df[lact_off_mid_cols].mean(axis=1).to_numpy()
high_5 = lact_off_df[lact_off_high_cols].mean(axis=1).to_numpy()

# =====================================================
# Moving Average (optional smoothing)
# =====================================================
def moving_average(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# =====================================================
# Helper Function
# =====================================================
def save_plot(filename):
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# =====================================================
# Figure 1 : Total Success
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(succ_1, label=label_1, color=COLOR[label_1])
plt.plot(succ_2, label=label_2, color=COLOR[label_2])
plt.plot(succ_3, label=label_3, color=COLOR[label_3])
plt.plot(succ_4, label=label_4, color=COLOR[label_4])
plt.plot(succ_5, label=label_5, color=COLOR[label_5])
plt.title("Total Successful Tasks per Training Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Number of Successful Tasks")
plt.legend()
save_plot("01_total_success.png")

# =====================================================
# Figure 2 : Total Failure
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(fail_1, label=label_1, color=COLOR[label_1])
plt.plot(fail_2, label=label_2, color=COLOR[label_2])
plt.plot(fail_3, label=label_3, color=COLOR[label_3])
plt.plot(fail_4, label=label_4, color=COLOR[label_4])
plt.plot(fail_5, label=label_5, color=COLOR[label_5])
plt.title("Total Failed Tasks per Training Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Number of Failed Tasks")
plt.legend()
save_plot("02_total_failure.png")

# =====================================================
# Figure 3 : Success Rate
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(rate_1 * 100, label=label_1, color=COLOR[label_1])
plt.plot(rate_2 * 100, label=label_2, color=COLOR[label_2])
plt.plot(rate_3 * 100, label=label_3, color=COLOR[label_3])
plt.plot(rate_4 * 100, label=label_4, color=COLOR[label_4])
plt.plot(rate_5 * 100, label=label_5, color=COLOR[label_5])
plt.title("Success Rate per Training Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Success Rate (%)")
plt.legend()
save_plot("03_success_rate.png")

# =====================================================
# Figure 4 : Failure Rate
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(fail_rate_1 * 100, label=label_1, color=COLOR[label_1])
plt.plot(fail_rate_2 * 100, label=label_2, color=COLOR[label_2])
plt.plot(fail_rate_3 * 100, label=label_3, color=COLOR[label_3])
plt.plot(fail_rate_4 * 100, label=label_4, color=COLOR[label_4])
plt.plot(fail_rate_5 * 100, label=label_5, color=COLOR[label_5])
plt.title("Failure Rate per Training Iteration")
plt.xlabel("Training Iteration")
plt.ylabel("Failure Rate (%)")
plt.legend()
save_plot("04_failure_rate.png")

# =====================================================
# Figure 5 : Success vs Failure, per mode
# =====================================================
for succ, fail, label in [
    (succ_1, fail_1, label_1),
    (succ_2, fail_2, label_2),
    (succ_3, fail_3, label_3),
    (succ_4, fail_4, label_4),
    (succ_5, fail_5, label_5),
]:
    plt.figure(figsize=(10, 5))
    plt.plot(succ, label="Success", color="tab:green")
    plt.plot(fail, label="Failure", color="tab:red")
    plt.title(f"{label}: Success vs Failure")
    plt.xlabel("Training Iteration")
    plt.ylabel("Number of Tasks")
    plt.legend()
    save_plot(f"05_{label}_success_vs_failure.png")

# =====================================================
# Figure 6 : Smoothed Success Rate
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(moving_average(rate_1 * 100), label=label_1, color=COLOR[label_1])
plt.plot(moving_average(rate_2 * 100), label=label_2, color=COLOR[label_2])
plt.plot(moving_average(rate_3 * 100), label=label_3, color=COLOR[label_3])
plt.plot(moving_average(rate_4 * 100), label=label_4, color=COLOR[label_4])
plt.plot(moving_average(rate_5 * 100), label=label_5, color=COLOR[label_5])
plt.title("Smoothed Success Rate")
plt.xlabel("Training Iteration")
plt.ylabel("Success Rate (%)")
plt.legend()
save_plot("06_smoothed_success_rate.png")

# =====================================================
# Figure 7 : Smoothed Failure Rate
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(moving_average(fail_rate_1 * 100), label=label_1, color=COLOR[label_1])
plt.plot(moving_average(fail_rate_2 * 100), label=label_2, color=COLOR[label_2])
plt.plot(moving_average(fail_rate_3 * 100), label=label_3, color=COLOR[label_3])
plt.plot(moving_average(fail_rate_4 * 100), label=label_4, color=COLOR[label_4])
plt.plot(moving_average(fail_rate_5 * 100), label=label_5, color=COLOR[label_5])
plt.title("Smoothed Failure Rate")
plt.xlabel("Training Iteration")
plt.ylabel("Failure Rate (%)")
plt.legend()
save_plot("07_smoothed_failure_rate.png")

# =====================================================
# Figures 8-10 : Completion Time comparison, one figure per priority,
# all 4 modes overlaid.
# =====================================================
for series, priority_name, fig_no in [
    ((low_1, low_2, low_3, low_4, low_5), "Low-Priority", "08"),
    ((mid_1, mid_2, mid_3, mid_4, mid_5), "Mid-Priority", "09"),
    ((high_1, high_2, high_3, high_4, high_5), "High-Priority", "10"),
]:
    s1, s2, s3, s4, s5 = series
    plt.figure(figsize=(10, 5))
    plt.plot(s1, label=label_1, color=COLOR[label_1])
    plt.plot(s2, label=label_2, color=COLOR[label_2])
    plt.plot(s3, label=label_3, color=COLOR[label_3])
    plt.plot(s4, label=label_4, color=COLOR[label_4])
    plt.plot(s5, label=label_5, color=COLOR[label_5])
    plt.yscale('log')  # preserves true magnitude differences; small series
                        # (e.g. LACT-Off) stay visible instead of collapsing
                        # to a flat line at 0, which a linear % scale can't avoid
    plt.title(f"{priority_name} Task Completion Time per Training Iteration")
    plt.xlabel("Training Iteration")
    plt.ylabel("Completion Time (s, log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    save_plot(f"{fig_no}_{priority_name.lower().replace('-', '_')}_completion_time.png")

# =====================================================
# Figure 11 : Summary grouped bar -- mean completion time per priority,
# per mode, on a log y-axis so both very small and very large means
# remain legible in the same chart.
# =====================================================
priorities = ["Low", "Mid", "High"]
means_by_label = {
    label_1: [np.mean(low_1), np.mean(mid_1), np.mean(high_1)],
    label_2: [np.mean(low_2), np.mean(mid_2), np.mean(high_2)],
    label_3: [np.mean(low_3), np.mean(mid_3), np.mean(high_3)],
    label_4: [np.mean(low_4), np.mean(mid_4), np.mean(high_4)],
    label_5: [np.mean(low_5), np.mean(mid_5), np.mean(high_5)],
}

x = np.arange(len(priorities))
n_labels = len(means_by_label)
bar_width = 0.8 / n_labels

fig, ax = plt.subplots(figsize=(10, 5))
for idx, (label, means) in enumerate(means_by_label.items()):
    offset = (idx - (n_labels - 1) / 2) * bar_width
    bars = ax.bar(x + offset, means, width=bar_width, label=label, color=COLOR[label])
    # annotate each bar with its actual value since log-scale bars are
    # hard to read precisely by eye, especially the very small ones
    for rect, val in zip(bars, means):
        ax.annotate(f"{val:.3g}", (rect.get_x() + rect.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=8, rotation=90)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(priorities)
ax.set_title("Mean Task Completion Time by Priority (log scale)")
ax.set_xlabel("Task Priority")
ax.set_ylabel("Mean Completion Time (s, log scale)")
ax.legend()
ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
save_plot("11_mean_completion_time_by_priority.png")

print(f"\nAll figures have been saved to '{OUTPUT_DIR}/'")