import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

NUM_SERVERS = cfg.NUM_SERVERS

label_1 = cfg.MODE_NAME[cfg.MINUS_PDMA]      # wide-format: "-PDMA"
label_2 = cfg.MODE_NAME[cfg.MINUS_LACT_Off]  # per-decision-file format
label_3 = cfg.MODE_NAME[cfg.PDMA]            # wide-format: "PDMA"
label_4 = cfg.MODE_NAME[cfg.LACT_Off_MINUS]  # per-decision-file format
label_5 = cfg.MODE_NAME[cfg.LACT_Off]        # per-decision-file format

OUTPUT_DIR = cfg.GRAPH_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

# ---- wide-format implementations (-PDMA, PDMA): one CSV each, with
# per-server "server_{i}_reward" / "server_{i}_critic_loss" columns -------
MINUS_PDMA_LOG = cfg.RESULT_DIR / f"{label_1}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
minus_pdma_df = pd.read_csv(MINUS_PDMA_LOG)

PDMA_LOG = cfg.RESULT_DIR / f"{label_3}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
pdma_df = pd.read_csv(PDMA_LOG)

WIDE_FORMAT_RUNS = [
    (label_1, minus_pdma_df, "tab:blue", "tab:green"),
    (label_3, pdma_df, "tab:purple", "tab:red"),
]

# ---- per-decision-file implementations (-LACT-Off, LACT-Off-, LACT-Off):
# one CSV per (server, decision), columns reward/actor_loss/critic_loss/
# shared_critic_loss --------------------------------------------------
def load_per_decision(label):
    return [
        [
            pd.read_csv(
                cfg.RESULT_DIR
                / f"{label}_TOPO[{cfg.CURRENT_TOPOLOGY}]training_log_server{server}_decision{d}.csv"
            )
            for d in range(3)
        ]
        for server in range(NUM_SERVERS)
    ]

minus_lactoff_df = load_per_decision(label_2)
lactoff_minus_df = load_per_decision(label_4)
lactoff_df = load_per_decision(label_5)

PER_DECISION_RUNS = [
    (label_2, minus_lactoff_df),
    (label_4, lactoff_minus_df),
    (label_5, lactoff_df),
]

# ==========================================================
# Draw graphs
# ==========================================================

for server in range(NUM_SERVERS):

    #########################################################
    # Figure set 1 : wide-format runs (-PDMA, PDMA)
    # Reward + Critic Loss only (no actor loss)
    #########################################################
    for label, df, reward_color, loss_color in WIDE_FORMAT_RUNS:
        reward_col = f"server_{server}_reward"
        critic_loss_col = f"server_{server}_critic_loss"

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axs[0].plot(df[reward_col], color=reward_color)
        axs[0].set_title(f"{label} : Server {server} Reward")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)

        axs[1].plot(df[critic_loss_col], color=loss_color)
        axs[1].set_title("Critic Loss")
        axs[1].set_xlabel("Training Iteration")
        axs[1].set_ylabel("Loss")
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"server_{server}_{label}.png"),
            dpi=300,
        )
        plt.close()

    #########################################################
    # Figure set 2 : per-decision-file runs
    # (-LACT-Off, LACT-Off-, LACT-Off)
    # Reward + Critic Loss + Shared Critic Loss (no actor loss)
    #########################################################
    for label, run_df in PER_DECISION_RUNS:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        for decision in range(3):
            df = run_df[server][decision]

            axs[0].plot(df["reward"], label=f"Decision {decision}")
            axs[1].plot(df["critic_loss"], label=f"Decision {decision}")
            axs[2].plot(df["shared_critic_loss"], label=f"Decision {decision}")

        axs[0].set_title(f"{label} : Server {server} Reward")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].set_title("Critic Loss")
        axs[1].set_ylabel("Loss")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].set_title("Shared Critic Loss")
        axs[2].set_xlabel("Training Iteration")
        axs[2].set_ylabel("Loss")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"server_{server}_{label}.png"),
            dpi=300,
        )
        plt.close()

print(f"All figures saved in '{OUTPUT_DIR}'")