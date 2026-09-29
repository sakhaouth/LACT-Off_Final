import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import config as cfg


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = cfg.RESULT_DIR
OUTPUT_FOLDER = cfg.GRAPH_DIR

CONFIDENCE_LEVEL = 0.95


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

METRICS = [
    "lstm_training_time",
    "lstm_train_loss",
    "lstm_test_loss",
    "lstm_test_mae",
    "llm_training_time",
    "llm_train_loss",
    "llm_test_loss",
    "llm_test_mae",
]


# ============================================================
# LINE STYLES
# ============================================================

LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (2, 2)),
]


# ============================================================
# FIND CSV FILES
# ============================================================

files = [
    os.path.join(
        DATA_FOLDER,
        f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]"
        f"train_predictor_log_bs{b}.csv"
    )
    for b in range(
        cfg.SERVER_COUNT[cfg.CURRENT_TOPOLOGY]
    )
]


print("\n========================================")
print("CSV FILES")
print("========================================")

for file in files:
    print(os.path.basename(file))


# ============================================================
# READ CSV FILES
# ============================================================

dataframes = []

for file in files:

    try:

        if not os.path.exists(file):

            print(
                f"WARNING: File not found: "
                f"{os.path.basename(file)}"
            )

            continue


        df = pd.read_csv(file)


        # ----------------------------------------------------
        # Find available metrics
        # ----------------------------------------------------

        available_metrics = [
            col
            for col in METRICS
            if col in df.columns
        ]


        if not available_metrics:

            print(
                f"WARNING: No required metrics found in "
                f"{os.path.basename(file)}"
            )

            continue


        # ----------------------------------------------------
        # Keep only required metrics
        # ----------------------------------------------------

        df = df[
            available_metrics
        ].copy()


        # ----------------------------------------------------
        # Convert values to numeric
        # ----------------------------------------------------

        for col in available_metrics:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        dataframes.append(df)


        print(
            f"Loaded: {os.path.basename(file)} "
            f"({len(df)} rows)"
        )


    except Exception as e:

        print(
            f"ERROR reading "
            f"{os.path.basename(file)}: {e}"
        )


# ============================================================
# CHECK DATA
# ============================================================

if not dataframes:

    raise RuntimeError(
        "No valid CSV files were loaded."
    )


print(
    f"\nTotal valid server files: "
    f"{len(dataframes)}"
)


# ============================================================
# CALCULATE STATISTICS ACROSS SERVERS
# ============================================================

statistics = {}


for metric in METRICS:

    series_list = []


    # --------------------------------------------------------
    # Collect metric from every server
    # --------------------------------------------------------

    for df in dataframes:

        if metric in df.columns:

            series_list.append(
                df[metric]
            )


    if not series_list:

        print(
            f"Skipping {metric}: "
            f"no data found."
        )

        continue


    # --------------------------------------------------------
    # Combine server data
    #
    # Each column = one server
    # Each row = same evaluation/training point
    # --------------------------------------------------------

    matrix = pd.concat(
        series_list,
        axis=1
    )


    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    mean = matrix.mean(
        axis=1,
        skipna=True
    )


    # --------------------------------------------------------
    # Sample standard deviation
    # --------------------------------------------------------

    sd = matrix.std(
        axis=1,
        skipna=True,
        ddof=1
    )


    # --------------------------------------------------------
    # Number of servers
    # --------------------------------------------------------

    n = matrix.count(
        axis=1
    )


    # --------------------------------------------------------
    # Standard error
    # --------------------------------------------------------

    se = (
        sd / np.sqrt(n)
    )


    # --------------------------------------------------------
    # Student's t critical value
    # --------------------------------------------------------

    t_value = pd.Series(
        np.nan,
        index=matrix.index,
        dtype=float
    )


    valid = n > 1


    t_value.loc[valid] = [

        stats.t.ppf(
            1 - (1 - CONFIDENCE_LEVEL) / 2,
            int(degrees_of_freedom)
        )

        for degrees_of_freedom
        in n.loc[valid]

    ]


    # --------------------------------------------------------
    # Confidence interval margin
    # --------------------------------------------------------

    margin = (
        t_value * se
    )


    # --------------------------------------------------------
    # 95% confidence interval
    # --------------------------------------------------------

    ci_lower = (
        mean - margin
    )

    ci_upper = (
        mean + margin
    )


    # --------------------------------------------------------
    # Store statistics
    # --------------------------------------------------------

    statistics[metric] = pd.DataFrame({

        "row": matrix.index,

        "n_servers": n,

        "mean": mean,

        "sd": sd,

        "ci_lower": ci_lower,

        "ci_upper": ci_upper,

        "ci_margin": margin,

    })


# ============================================================
# SAVE STATISTICS
# ============================================================

print("\n========================================")
print("SAVING STATISTICS")
print("========================================")


# for metric, result in statistics.items():

#     output_file = os.path.join(
#         OUTPUT_FOLDER,
#         f"{metric}_statistics.csv"
#     )


#     result.to_csv(
#         output_file,
#         index=False
#     )


#     print(
#         f"Saved: {output_file}"
#     )


# ============================================================
# FUNCTION 1
#
# RAW METRIC GRAPH
#
# Used for:
#   1. Training time
#   2. Prediction performance
#
# Shows:
#   - Mean
#   - SD
#   - 95% CI
# ============================================================

def plot_metrics(
    metrics,
    title,
    output_filename,
    y_title,
    x_title
):

    plt.figure(
        figsize=(16, 9)
    )


    plotted_any = False


    # ========================================================
    # Plot each metric
    # ========================================================

    for i, metric in enumerate(metrics):


        if metric not in statistics:

            print(
                f"WARNING: {metric} "
                f"not available."
            )

            continue


        result = statistics[
            metric
        ].copy()


        # Remove rows without mean

        result = result.dropna(
            subset=["mean"]
        )


        if result.empty:

            continue


        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        x = result["row"]


        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        mean = result[
            "mean"
        ]


        # ----------------------------------------------------
        # Standard deviation
        # ----------------------------------------------------

        sd = result[
            "sd"
        ]


        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        ci_lower = result[
            "ci_lower"
        ]

        ci_upper = result[
            "ci_upper"
        ]


        # ----------------------------------------------------
        # Line style
        # ----------------------------------------------------

        line_style = (
            LINE_STYLES[
                i % len(LINE_STYLES)
            ]
        )


        # ====================================================
        # Mean line
        # ====================================================

        plt.plot(
            x,
            mean,
            linestyle=line_style,
            linewidth=2,
            label=metric
        )


        # ====================================================
        # 95% CI shaded area
        # ====================================================

        valid_ci = (
            ci_lower.notna()
            &
            ci_upper.notna()
        )


        if valid_ci.any():

            plt.fill_between(

                x[valid_ci],

                ci_lower[valid_ci],

                ci_upper[valid_ci],

                alpha=0.08

            )


        # ====================================================
        # SD error bars
        # ====================================================

        valid_sd = sd.notna()


        if valid_sd.any():

            valid_indices = (
                result.index[
                    valid_sd
                ]
            )


            # Approximately 20 error bars

            step = max(
                1,
                len(valid_indices) // 20
            )


            selected_indices = (
                valid_indices[::step]
            )


            plt.errorbar(

                result.loc[
                    selected_indices,
                    "row"
                ],

                result.loc[
                    selected_indices,
                    "mean"
                ],

                yerr=result.loc[
                    selected_indices,
                    "sd"
                ],

                fmt="none",

                capsize=3,

                alpha=0.4

            )
            plt.yscale("log")


        plotted_any = True


    # ========================================================
    # Check plotting
    # ========================================================

    if not plotted_any:

        plt.close()

        print(
            f"WARNING: Nothing to plot for "
            f"{output_filename}"
        )

        return


    # ========================================================
    # Formatting
    # ========================================================

    plt.xlabel(
        x_title,
        fontsize=12
    )


    plt.ylabel(
        y_title,
        fontsize=12
    )


    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )


    plt.grid(
        True,
        alpha=0.25
    )


    plt.legend(
        fontsize=10,
        loc="best"
    )


    plt.tight_layout()


    # ========================================================
    # Save
    # ========================================================

    output_file = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved graph: {output_file}"
    )


# ============================================================
# FUNCTION 2
#
# MIN-MAX NORMALIZED LLM TEST SUBPLOTS
#
# Subplot 1:
#   LLM Test Loss
#
# Subplot 2:
#   LLM Test MAE
#
# Shows:
#   - Normalized mean only
#
# Does NOT show:
#   - SD
#   - CI
# ============================================================

def plot_normalized_llm_test_subplots(
    title,
    output_filename
):

    # --------------------------------------------------------
    # Create two horizontal subplots
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )


    metrics = [

        (
            "llm_test_loss",
            "LLM Test Loss"
        ),

        (
            "llm_test_mae",
            "LLM Test MAE"
        ),

    ]


    # ========================================================
    # Process each metric
    # ========================================================

    for i, (
        metric,
        subplot_title
    ) in enumerate(metrics):


        ax = axes[i]


        # ----------------------------------------------------
        # Check metric
        # ----------------------------------------------------

        if metric not in statistics:

            ax.set_title(
                subplot_title
            )

            ax.text(

                0.5,
                0.5,

                "Data not available",

                ha="center",
                va="center",

                transform=ax.transAxes

            )

            continue


        # ----------------------------------------------------
        # Get result
        # ----------------------------------------------------

        result = statistics[
            metric
        ].copy()


        result = result.dropna(
            subset=["mean"]
        )


        if result.empty:

            ax.set_title(
                subplot_title
            )

            ax.text(

                0.5,
                0.5,

                "No valid data",

                ha="center",
                va="center",

                transform=ax.transAxes

            )

            continue


        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        x = result[
            "row"
        ]


        # ----------------------------------------------------
        # Original mean
        # ----------------------------------------------------

        mean = result[
            "mean"
        ].copy()


        # ====================================================
        # MIN-MAX NORMALIZATION
        #
        # normalized =
        #
        # (x - min(x))
        # ----------------
        # (max(x) - min(x))
        #
        # Minimum = 0
        # Maximum = 1
        # ====================================================

        min_value = mean.min()

        max_value = mean.max()


        # ----------------------------------------------------
        # Check if all values are identical
        # ----------------------------------------------------

        if (
            max_value == min_value
        ):

            # If all values are the same,
            # there is no range to normalize.

            normalized_mean = (
                np.zeros(
                    len(mean)
                )
            )

        else:

            normalized_mean = (
                (mean - min_value)
                /
                (max_value - min_value)
            )


        # ====================================================
        # Plot normalized mean
        # ====================================================

        ax.plot(

            x,

            normalized_mean,

            linestyle=LINE_STYLES[i],

            linewidth=2,

            label=metric

        )


        # ====================================================
        # Subplot title
        # ====================================================

        ax.set_title(

            subplot_title,

            fontsize=14,

            fontweight="bold"

        )


        # ====================================================
        # X axis
        # ====================================================

        ax.set_xlabel(

            "Training Row / Evaluation Point",

            fontsize=11

        )


        # ====================================================
        # Y axis
        # ====================================================

        ax.set_ylabel(

            "Min-Max Normalized Value",

            fontsize=11

        )


        # ====================================================
        # Y-axis range
        # ====================================================

        ax.set_ylim(
            0,
            1.05
        )


        # ====================================================
        # Grid
        # ====================================================

        ax.grid(
            True,
            alpha=0.25
        )


        # ====================================================
        # Legend
        # ====================================================

        ax.legend(
            fontsize=10,
            loc="best"
        )


    # ========================================================
    # Overall figure title
    # ========================================================

    fig.suptitle(

        title,

        fontsize=16,

        fontweight="bold"

    )


    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94
        ]
    )


    # ========================================================
    # Save graph
    # ========================================================

    output_file = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )


    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    print(
        f"Saved normalized subplot graph: "
        f"{output_file}"
    )


# ============================================================
# GRAPH 1
#
# LSTM VS LLM TRAINING TIME
#
# Mean + SD + 95% CI
# ============================================================

plot_metrics(

    metrics=[

        "lstm_training_time",

        "llm_training_time",

    ],

    title=(
        "LSTM vs LLM Training Time"
    ),

    output_filename=(
        "training_time_comparison.png"
    ),
    y_title="Training Time(Log Scale)",
    x_title="Epoch"

)


# ============================================================
# GRAPH 2
#
# LSTM AND LLM PREDICTION PERFORMANCE
#
# Mean + SD + 95% CI
# ============================================================

plot_metrics(

    metrics=[

        "lstm_train_loss",

        "lstm_test_loss",

        "lstm_test_mae",

        "llm_train_loss",

        "llm_test_loss",

        "llm_test_mae",

    ],

    title=(
        "LSTM and LLM Prediction Performance"
    ),

    output_filename=(
        "prediction_performance_comparison.png"
    ),
    y_title="Loss(Log Scale)",
    x_title="Epoch"

)


# ============================================================
# GRAPH 3
#
# MIN-MAX NORMALIZED LLM TEST PERFORMANCE
#
# Two subplots:
#
#   1. LLM Test Loss
#   2. LLM Test MAE
#
# Mean only
# No SD
# No CI
# =================================================