import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
import os

# ---------------------------------------------------------
# 1. Provide your dataframes and algorithm names
# ---------------------------------------------------------

# Example:
dataframes = []
algorithm_names = []

# Each dataframe must contain the columns described in your question.

PERFORMANCE = 0
SENSITIVITY = 1
TEMPORARY = 2
HOMO = 3
GRAPH_MODE = HOMO

NUM_SERVERS = cfg.NUM_SERVERS

label_1 = cfg.MODE_NAME[cfg.MINUS_PDMA]
label_2 = cfg.MODE_NAME[cfg.MINUS_LACT_Off]
label_3 = cfg.MODE_NAME[cfg.PDMA]
label_4 = cfg.MODE_NAME[cfg.LACT_Off_MINUS]
label_5 = cfg.MODE_NAME[cfg.LACT_Off]

lact_ogf_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]  / cfg.MODE_NAME[cfg.LACT_Off]  / f"[SENSITIVITY{cfg.CPU_SENSITIVITY}_{cfg.MEM_SENSITIVITY}_{cfg.SIZE_SENSITIVITY}]_{cfg.MODE_NAME[cfg.LACT_Off]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
pdma_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]  / cfg.MODE_NAME[cfg.PDMA] / f"[SENSITIVITY{cfg.CPU_SENSITIVITY}_{cfg.MEM_SENSITIVITY}_{cfg.SIZE_SENSITIVITY}]_{cfg.MODE_NAME[cfg.PDMA]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"

if GRAPH_MODE == PERFORMANCE:
    lact_off_minus_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]  / cfg.MODE_NAME[cfg.LACT_Off_MINUS]  / f"[SENSITIVITY{cfg.CPU_SENSITIVITY}_{cfg.MEM_SENSITIVITY}_{cfg.SIZE_SENSITIVITY}]_{cfg.MODE_NAME[cfg.LACT_Off_MINUS]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
    minus_lact_off_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]  / cfg.MODE_NAME[cfg.MINUS_LACT_Off]  / f"[SENSITIVITY{cfg.CPU_SENSITIVITY}_{cfg.MEM_SENSITIVITY}_{cfg.SIZE_SENSITIVITY}]_{cfg.MODE_NAME[cfg.MINUS_LACT_Off]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker0.csv"
    minus_pdma_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]  / cfg.MODE_NAME[cfg.MINUS_PDMA] / f"[SENSITIVITY{cfg.CPU_SENSITIVITY}_{cfg.MEM_SENSITIVITY}_{cfg.SIZE_SENSITIVITY}]_{cfg.MODE_NAME[cfg.MINUS_PDMA]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"

    dataframes.extend([pd.read_csv(minus_pdma_dir), pd.read_csv(minus_lact_off_dir), pd.read_csv(pdma_dir), pd.read_csv(lact_off_minus_dir), pd.read_csv(lact_ogf_dir)])
    algorithm_names.extend([cfg.MODE_DES[cfg.MINUS_PDMA], cfg.MODE_DES[cfg.MINUS_LACT_Off], cfg.MODE_DES[cfg.PDMA], cfg.MODE_DES[cfg.LACT_Off_MINUS], cfg.MODE_DES[cfg.LACT_Off]])
if GRAPH_MODE == SENSITIVITY:
    dataframes.extend([pd.read_csv(pdma_dir), pd.read_csv(lact_ogf_dir)])
    algorithm_names.extend([cfg.MODE_DES[cfg.PDMA], cfg.MODE_DES[cfg.LACT_Off]])
# if GRAPH_MODE == TEMPORARY:
#     dataframes.extend([pd.read_csv(minus_pdma_dir), pd.read_csv(minus_lact_off_dir), pd.read_csv(pdma_dir), pd.read_csv(lact_ogf_dir)])
#     algorithm_names.extend([cfg.MODE_DES[cfg.MINUS_PDMA], cfg.MODE_DES[cfg.MINUS_LACT_Off], cfg.MODE_DES[cfg.PDMA], cfg.MODE_DES[cfg.LACT_Off]])
if GRAPH_MODE == HOMO:
    lact_ogf_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.TIER_6]  / cfg.MODE_NAME[cfg.LACT_Off]  / f"[SENSITIVITY{0}_{0}_{0}]_{cfg.MODE_NAME[cfg.LACT_Off]}_TOPO[{cfg.TIER_6}]_performance_parameter_worker0.csv"
    pdma_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.TIER_6]  / cfg.MODE_NAME[cfg.PDMA] / f"[SENSITIVITY{0}_{0}_{0}]_{cfg.MODE_NAME[cfg.PDMA]}_TOPO[{cfg.TIER_6}]_training_log.csv"

    lact_off_homo_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.TIER_6]  / cfg.MODE_NAME[cfg.LACT_Off]  / f"[HOMO][SENSITIVITY{0}_{0}_{0}]_{cfg.MODE_NAME[cfg.LACT_Off]}_TOPO[{cfg.TIER_6}]_performance_parameter_worker0.csv"
  
    pdma_homo_dir = cfg.RESULT_DIR / cfg.TOPO_NAME[cfg.TIER_6]  / cfg.MODE_NAME[cfg.PDMA] / f"[HOMO][SENSITIVITY{0}_{0}_{0}]_{cfg.MODE_NAME[cfg.PDMA]}_TOPO[{cfg.TIER_6}]_training_log.csv"
    dataframes.extend([pd.read_csv(pdma_homo_dir), pd.read_csv(pdma_dir), pd.read_csv(lact_off_homo_dir), pd.read_csv(lact_ogf_dir)])
    algorithm_names.extend(["PDMA_HOMO", "PDMA_HETE", "LACT-Off_HOMO", "LACT-Off_HETE"])


assert len(dataframes) == len(algorithm_names), (
    "Each dataframe must have a corresponding algorithm name."
)

# Six task groups: 0, 1, 2, 3, 4, 5
groups = range(NUM_SERVERS)

success_cols = [f"success_{i}" for i in groups]
timeout_cols = [f"time_out_{i}" for i in groups]
generation_cols = [f"generation_{i}" for i in groups]

# ---------------------------------------------------------
# 2. Calculate total successful tasks, timeouts, and
#    generated tasks for each algorithm
# ---------------------------------------------------------
OUTPUT_DIR = cfg.GRAPH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
def save_plot(plt, filename):
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, str(GRAPH_MODE)+filename), dpi=300)
    plt.close()
total_success = []
total_timeout = []
total_generation = []
success_rates = []
COLOR = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
}
for df in dataframes:

    # Convert columns to numeric, treating invalid values as NaN.
    success = df[success_cols].apply(pd.to_numeric, errors="coerce")
    timeout = df[timeout_cols].apply(pd.to_numeric, errors="coerce")

    generation = df[generation_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    
    # Total values across all episodes and task groups.
    success_sum = success.sum(axis = 1)
    # print(success.sum(axis = 1))
    timeout_sum = timeout.sum(axis = 1)
    generation_sum = generation.sum(axis = 1)

    total_success.append(success_sum)
    total_timeout.append(timeout_sum)
    total_generation.append(generation_sum)
    # print(success_sum)
    # print(generation_sum)
    # Overall success rate.
    rate = success_sum / generation_sum * 100

    success_rates.append(rate)


# ---------------------------------------------------------
# 3. Graph 1: Total successful tasks
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

# plt.bar(algorithm_names, total_success)

for x in range(len(algorithm_names)):
    plt.plot(total_success[x], label=algorithm_names[x], color=COLOR[x])

plt.xlabel("Algorithm")
plt.ylabel("Total Successful Tasks")
plt.title("Total Successful Tasks by framework")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.legend()
save_plot(plt, "Total_Success.png")
# plt.show()



# ---------------------------------------------------------
# 4. Graph 2: Total timed-out tasks
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

# plt.bar(algorithm_names, total_timeout)
for x in range(len(algorithm_names)):
    plt.plot(total_timeout[x], label=algorithm_names[x], color=COLOR[x])
plt.xlabel("Algorithm")
plt.ylabel("Total Timed-Out Tasks")
plt.title("Total Timed-Out Tasks by Algorithm")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.legend()
# plt.show()
save_plot(plt, "Total_Time.png")


# ---------------------------------------------------------
# 5. Graph 3: Overall task success rate
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

# plt.bar(algorithm_names, success_rates)

for x in range(len(algorithm_names)):
    plt.plot(success_rates[x], label=algorithm_names[x], color=COLOR[x])
plt.xlabel("Algorithm")
plt.ylabel("Success Rate (%)")
plt.title("Overall Task Success Rate by Algorithm")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
# plt.show()
save_plot(plt, "Success Rate.png")

# ---------------------------------------------------------
# 6. Graph 4: Average execution time by task category
# ---------------------------------------------------------
# ASSUMPTION:
# low-0 ... low-5, mid-0 ... mid-5, and high-0 ... high-5
# contain execution-time measurements.
#
# If these columns contain task counts instead, you will
# need the corresponding execution-time measurements.

categories = ["low_time", "mid_time", "high_time"]

average_times = {category: [] for category in categories}

for df in dataframes:

    for category in categories:

        cols = [f"{category}_{i}" for i in groups]

        values = df[cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        # Ignore NaN values; return NaN if no valid values exist.
        valid_values = values[np.isfinite(values)]

        avg_time = (
            np.mean(valid_values)
            if valid_values.size > 0
            else np.nan
        )

        average_times[category].append(avg_time)


# Grouped bar chart
x = np.arange(len(algorithm_names))
width = 0.25

plt.figure(figsize=(12, 6))

for i, category in enumerate(categories):

    positions = x + (i - 1) * width

    plt.bar(
        positions,
        average_times[category],
        width=width,
        label=category.capitalize()
    )

plt.xlabel("Algorithm")
plt.ylabel("Average Task Execution Time")
plt.title("Average Task Execution Time by Category and Algorithm")
plt.xticks(x, algorithm_names, rotation=30, ha="right")
plt.legend(title="Task Category")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
# plt.show()
save_plot(plt, "Task_Execution_Time.png")