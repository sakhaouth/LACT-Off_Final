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
# Helpers
# ==========================================================

def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Falls back to zeros if constant."""
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0
    return (series - lo) / (hi - lo)


def average_across_servers(series_list):
    """Average a list of same-length pandas Series (one per server)."""
    return pd.concat(series_list, axis=1).mean(axis=1)


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

#############################################################
# Wide-format runs (-PDMA, PDMA)
# Graph 1: avg normalized reward across servers (1 line)
# Graph 2: avg normalized critic loss across servers (1 line)
#############################################################
for label, df, reward_color, loss_color in WIDE_FORMAT_RUNS:

    reward_series = [normalize(df[f"server_{s}_reward"]) for s in range(NUM_SERVERS)]
    critic_series = [normalize(df[f"server_{s}_critic_loss"]) for s in range(NUM_SERVERS)]

    avg_reward = average_across_servers(reward_series)
    avg_critic_loss = average_across_servers(critic_series)

    # ---- Reward graph ----
    plt.figure(figsize=(12, 5))
    plt.plot(avg_reward, color=reward_color)
    plt.title(f"{label} : Average Normalized Reward (all servers)")
    plt.xlabel("Training Iteration")
    plt.ylabel("Normalized Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_avg_reward.png"), dpi=300)
    plt.close()

    # ---- Critic loss graph ----
    plt.figure(figsize=(12, 5))
    plt.plot(avg_critic_loss, color=loss_color)
    plt.title(f"{label} : Average Normalized Critic Loss (all servers)")
    plt.xlabel("Training Iteration")
    plt.ylabel("Normalized Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_avg_critic_loss.png"), dpi=300)
    plt.close()

#############################################################
# Per-decision-file runs (-LACT-Off, LACT-Off-, LACT-Off)
# Graph 1: avg normalized reward across servers, 3 lines (one per decision)
# Graph 2: avg normalized critic loss across servers, 4 lines
#          (3 decision critic losses + 1 shared critic loss)
#############################################################
for label, run_df in PER_DECISION_RUNS:

    # ---- Reward graph : 3 lines ----
    plt.figure(figsize=(12, 5))
    for decision in range(3):
        reward_series = [
            normalize(run_df[server][decision]["reward"])
            for server in range(NUM_SERVERS)
        ]
        avg_reward = average_across_servers(reward_series)
        plt.plot(avg_reward, label=f"Decision {decision}")

    plt.title(f"{label} : Average Normalized Reward (all servers)")
    plt.xlabel("Training Iteration")
    plt.ylabel("Normalized Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_avg_reward.png"), dpi=300)
    plt.close()

    # ---- Critic loss graph : 4 lines (3 critic_loss + 1 shared_critic_loss) ----
    plt.figure(figsize=(12, 5))
    for decision in range(3):
        critic_series = [
            normalize(run_df[server][decision]["critic_loss"])
            for server in range(NUM_SERVERS)
        ]
        avg_critic_loss = average_across_servers(critic_series)
        plt.plot(avg_critic_loss, label=f"Critic Loss (Decision {decision})")

    # Shared critic loss: average across servers AND decisions into one line
    shared_series = [
        normalize(run_df[server][decision]["shared_critic_loss"])
        for server in range(NUM_SERVERS)
        for decision in range(3)
    ]
    avg_shared_critic_loss = average_across_servers(shared_series)
    plt.plot(avg_shared_critic_loss, label="Shared Critic Loss", linestyle="--", color="black")

    plt.title(f"{label} : Average Normalized Critic Loss (all servers)")
    plt.xlabel("Training Iteration")
    plt.ylabel("Normalized Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{label}_avg_critic_loss.png"), dpi=300)
    plt.close()

print(f"All figures saved in '{OUTPUT_DIR}'")