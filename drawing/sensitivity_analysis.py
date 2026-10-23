import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

NUM_TASK_TYPES = 3


def plot_algorithm_comparison(
    sensitivity_data,
    sensitivity_values=(0, 1, 2),
    algorithm_names=None,
    save_dir=None,
):
    """
    Generate 6 graphs:
      - 3 execution-time graphs, one per sensitivity type.
      - 3 completion-rate graphs, one per sensitivity type.

    Each graph shows all three task types for all algorithms.

    Parameters
    ----------
    sensitivity_data : list[list[pd.DataFrame]]
        sensitivity_data[s][a] is the DataFrame for algorithm a
        under sensitivity type s.

        Example:
        [
            [random_s0, maddpg_s0, algorithm3_s0],
            [random_s1, maddpg_s1, algorithm3_s1],
            [random_s2, maddpg_s2, algorithm3_s2],
        ]

    sensitivity_values : sequence
        Sensitivity values to show on the x-axis.

    algorithm_names : list[str], optional
        Names of the algorithms.

    save_dir : str or Path, optional
        Directory for saving the figures.

    Notes
    -----
    Assumes:
      - success_0, success_1, success_2 represent completion indicators
        or per-episode completion rates.
      - generation_0, generation_1, generation_2 contain execution times.
    """

    P = len(sensitivity_data)

    if P == 0:
        raise ValueError("sensitivity_data is empty.")

    if len(sensitivity_values) != P:
        raise ValueError(
            "The number of sensitivity values must match "
            "the number of sensitivity groups."
        )

    N = len(sensitivity_data[0])

    if N == 0:
        raise ValueError("No algorithm DataFrames were provided.")

    if any(len(group) != N for group in sensitivity_data):
        raise ValueError(
            "Each sensitivity group must contain the same number "
            "of algorithm DataFrames."
        )

    if algorithm_names is None:
        algorithm_names = [
            f"Algorithm {i + 1}" for i in range(N)
        ]

    if len(algorithm_names) != N:
        raise ValueError(
            "The number of algorithm names must equal N."
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # results[metric][task][algorithm] = values across sensitivities
    results = {
        metric: {
            task: {
                algorithm: []
                for algorithm in range(N)
            }
            for task in range(NUM_TASK_TYPES)
        }
        for metric in ("completion_rate", "execution_time")
    }

    # --------------------------------------------------
    # Calculate the average metrics
    # --------------------------------------------------

    for s, algorithm_dfs in enumerate(sensitivity_data):

        for a, df in enumerate(algorithm_dfs):

            for task in range(NUM_TASK_TYPES):

                success_col = f"success_{task}"
                time_col = f"generation_{task}"

                missing = [
                    col for col in (success_col, time_col)
                    if col not in df.columns
                ]

                if missing:
                    raise ValueError(
                        f"Missing columns {missing} for "
                        f"sensitivity={sensitivity_values[s]}, "
                        f"algorithm={algorithm_names[a]}."
                    )

                # Average completion rate
                success = pd.to_numeric(
                    df[success_col], errors="coerce"
                ).to_numpy(dtype=float)

                valid_success = success[np.isfinite(success)]

                avg_completion_rate = (
                    np.mean(valid_success)
                    if valid_success.size > 0
                    else np.nan
                )

                results["completion_rate"][task][a].append(
                    avg_completion_rate
                )

                # Average execution time
                execution_time = pd.to_numeric(
                    df[time_col], errors="coerce"
                ).to_numpy(dtype=float)

                valid_times = execution_time[
                    np.isfinite(execution_time)
                ]

                avg_execution_time = (
                    np.mean(valid_times)
                    if valid_times.size > 0
                    else np.nan
                )

                results["execution_time"][task][a].append(
                    avg_execution_time
                )

    # --------------------------------------------------
    # Generate 6 graphs:
    # 3 sensitivity types x 2 performance metrics.
    #
    # Each graph has:
    #   - X-axis: sensitivity values
    #   - Lines: algorithm-task combinations
    # --------------------------------------------------

    metrics_to_plot = [
        (
            "execution_time",
            "Average Task Execution Time",
            "Average Execution Time",
        ),
        (
            "completion_rate",
            "Average Task Completion Rate",
            "Average Completion Rate",
        ),
    ]

    for metric_key, title, ylabel in metrics_to_plot:

        for s in range(P):

            fig, ax = plt.subplots(figsize=(11, 6))

            # Each line corresponds to one algorithm-task combination.
            for task in range(NUM_TASK_TYPES):

                for a, algorithm_name in enumerate(algorithm_names):

                    y = results[metric_key][task][a]

                    ax.plot(
                        sensitivity_values,
                        y,
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        label=f"{algorithm_name} - Task {task}",
                    )

            ax.set_title(
                f"{title} — Sensitivity Type {s}"
            )
            ax.set_xlabel("Sensitivity Value")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sensitivity_values)
            ax.grid(True, alpha=0.3)

            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=9,
            )

            fig.tight_layout()

            if save_dir is not None:
                filename = (
                    f"{metric_key}_sensitivity_type_{s}.png"
                )

                fig.savefig(
                    save_dir / filename,
                    dpi=300,
                    bbox_inches="tight",
                )

            plt.show()
            plt.close(fig)

    return results