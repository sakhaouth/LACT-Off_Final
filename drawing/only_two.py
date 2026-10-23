"""Compare only label_3 (PDMA) and label_5 (LACT_Off)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

NUM_SERVERS = cfg.NUM_SERVERS
label_3 = cfg.MODE_NAME[cfg.PDMA]
label_5 = cfg.MODE_NAME[cfg.LACT_Off]

PDMA_LOG = cfg.RESULT_DIR / f"[SENSITIVITY]_{label_3}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
LACT_Off_LOG = cfg.RESULT_DIR / f"[SENSITIVITY]_{label_5}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
OUTPUT_DIR = cfg.GRAPH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR = {label_3: "tab:green", label_5: "tab:purple"}
EPS = 1e-8

# Read the two different CSV formats.
pdma_df = pd.read_csv(PDMA_LOG)
lact_off_df = pd.read_csv(LACT_Off_LOG)

# PDMA uses *_training_log.csv column names.
pdma_success_cols = [f"server_{i}_total_success" for i in range(NUM_SERVERS)]
pdma_failure_cols = [f"server_{i}_total_violation" for i in range(NUM_SERVERS)]
pdma_low_cols = [f"low-{i}" for i in range(NUM_SERVERS)]
pdma_mid_cols = [f"mid-{i}" for i in range(NUM_SERVERS)]
pdma_high_cols = [f"high-{i}" for i in range(NUM_SERVERS)]

# LACT_Off uses *_performance_parameter_worker0.csv column names.
lact_success_cols = [f"success_{i}" for i in range(NUM_SERVERS)]
lact_failure_cols = [f"time_out_{i}" for i in range(NUM_SERVERS)]
lact_low_cols = [f"low_time_{i}" for i in range(NUM_SERVERS)]
lact_mid_cols = [f"mid_time_{i}" for i in range(NUM_SERVERS)]
lact_high_cols = [f"high_time_{i}" for i in range(NUM_SERVERS)]

# Check required columns before plotting.
for path, df, columns in [
    (PDMA_LOG, pdma_df, pdma_success_cols + pdma_failure_cols + pdma_low_cols + pdma_mid_cols + pdma_high_cols),
    (LACT_Off_LOG, lact_off_df, lact_success_cols + lact_failure_cols + lact_low_cols + lact_mid_cols + lact_high_cols),
]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")

# Total successes and failures per iteration.
succ_3 = pdma_df[pdma_success_cols].sum(axis=1).to_numpy()
fail_3 = pdma_df[pdma_failure_cols].sum(axis=1).to_numpy()
succ_5 = lact_off_df[lact_success_cols].sum(axis=1).to_numpy()
fail_5 = lact_off_df[lact_failure_cols].sum(axis=1).to_numpy()

# Rates use each mode's own success/failure counts.
rate_3 = succ_3 / (succ_3 + fail_3 + EPS)
rate_5 = succ_5 / (succ_5 + fail_5 + EPS)
fail_rate_3 = fail_3 / (succ_3 + fail_3 + EPS)
fail_rate_5 = fail_5 / (succ_5 + fail_5 + EPS)

# Average completion time across servers for each priority.
low_3 = pdma_df[pdma_low_cols].mean(axis=1).to_numpy()
mid_3 = pdma_df[pdma_mid_cols].mean(axis=1).to_numpy()
high_3 = pdma_df[pdma_high_cols].mean(axis=1).to_numpy()
low_5 = lact_off_df[lact_low_cols].mean(axis=1).to_numpy()
mid_5 = lact_off_df[lact_mid_cols].mean(axis=1).to_numpy()
high_5 = lact_off_df[lact_high_cols].mean(axis=1).to_numpy()


def moving_average(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_plot(filename):
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_comparison(y3, y5, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(y3, label=label_3, color=COLOR[label_3])
    plt.plot(y5, label=label_5, color=COLOR[label_5])
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(filename)


# Figure 1: total successful tasks.
plot_comparison(succ_3, succ_5, "Total Successful Tasks per Training Iteration",
                "Number of Successful Tasks", "[ONLY]_01_total_success.png")

# Figure 2: total failed tasks.
plot_comparison(fail_3, fail_5, "Total Failed Tasks per Training Iteration",
                "Number of Failed Tasks", "[ONLY]_02_total_failure.png")

# Figure 3: success rate (%).
plot_comparison(rate_3 * 100, rate_5 * 100, "Success Rate per Training Iteration",
                "Success Rate (%)", "[ONLY]_03_success_rate.png")

# Figure 4: failure rate (%).
plot_comparison(fail_rate_3 * 100, fail_rate_5 * 100, "Failure Rate per Training Iteration",
                "Failure Rate (%)", "[ONLY]_04_failure_rate.png")

# Figure 5: success versus failure for each mode.
for success, failure, label, filename in [
    (succ_3, fail_3, label_3, "[ONLY]_05_label3_success_vs_failure.png"),
    (succ_5, fail_5, label_5, "[ONLY]_05_label5_success_vs_failure.png"),
]:
    plt.figure(figsize=(10, 5))
    plt.plot(success, label="Success", color="tab:green")
    plt.plot(failure, label="Failure", color="tab:red")
    plt.title(f"{label}: Success vs Failure")
    plt.xlabel("Training Iteration")
    plt.ylabel("Number of Tasks")
    plt.legend()
    save_plot(filename)

# Figure 6: 10-iteration moving-average success rate.
plot_comparison(moving_average(rate_3 * 100), moving_average(rate_5 * 100),
                "Smoothed Success Rate (10-Iteration Moving Average)",
                "Success Rate (%)", "06_smoothed_success_rate.png")

# Figure 7: 10-iteration moving-average failure rate.
plot_comparison(moving_average(fail_rate_3 * 100), moving_average(fail_rate_5 * 100),
                "Smoothed Failure Rate (10-Iteration Moving Average)",
                "Failure Rate (%)", "07_smoothed_failure_rate.png")

# Figures 8-10: completion time by priority (linear y-axis).
for values_3, values_5, priority, number in [
    (low_3, low_5, "Low-Priority", "08"),
    (mid_3, mid_5, "Mid-Priority", "09"),
    (high_3, high_5, "High-Priority", "10"),
]:
    plt.figure(figsize=(10, 5))
    plt.plot(values_3, label=label_3, color=COLOR[label_3])
    plt.plot(values_5, label=label_5, color=COLOR[label_5])
    plt.title(f"{priority} Task Completion Time per Training Iteration")
    plt.xlabel("Training Iteration")
    plt.ylabel("Completion Time (second)")
    plt.legend()
    save_plot(f"[ONLY]_{number}_{priority.lower().replace('-', '_')}_completion_time.png")

# Figure 11: mean completion time by priority.
priorities = ["Low", "Mid", "High"]
means_by_label = {
    label_3: [np.mean(low_3), np.mean(mid_3), np.mean(high_3)],
    label_5: [np.mean(low_5), np.mean(mid_5), np.mean(high_5)],
}
x = np.arange(len(priorities))
bar_width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
for index, (label, means) in enumerate(means_by_label.items()):
    offset = (index - 0.5) * bar_width
    bars = ax.bar(x + offset, means, width=bar_width, label=label, color=COLOR[label])
    for bar, value in zip(bars, means):
        ax.annotate(f"{value:.3g}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=8, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(priorities)
ax.set_title("Mean Task Completion Time by Priority")
ax.set_xlabel("Task Priority")
ax.set_ylabel("Mean Completion Time (second)")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
save_plot("[ONLY]_11_mean_completion_time_by_priority.png")

print(f"\nAll figures for {label_3} and {label_5} saved to '{OUTPUT_DIR}/'")
