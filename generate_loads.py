"""
End-to-end training entry point.

Usage (illustrative -- CONFUSION: no CLI spec given, kept minimal):
    python -m training.train --pretrain_vrnn --episodes 1000
"""
import argparse
import numpy as np

from madrl_edge.core.fixed_environment import MultiAgentEdgeEnv
from madrl_edge.core.traffic import generate_slot_arrivals
from madrl_edge.agents.fixed_maddpg_agent import MADDPGCoordinator
from madrl_edge.agents.replay_buffer import MultiAgentReplayBuffer
from madrl_edge.models.new_vrnnn import FrozenVRNNEncoder, train_vrnn
from datetime import datetime, timedelta
import pandas as pd
import config as cfg
import time

# from utils import losses

# from utils import losses

# FIX: the RunningRewardNormalizer that used to live here has been removed.
# It was normalizing reward_n and pushing the NORMALIZED value into the
# replay buffer, while MADDPGCoordinator.update() (fixed_maddpg_agent.py)
# ALSO runs its own RunningNorm over whatever it samples back out of the
# buffer. That's reward normalization applied twice, with two different
# running-statistics estimators computed at two different times (once at
# push-time with the stats known back then, once at sample-time with
# whatever stats have accumulated since) -- so the same stored transition's
# effective TD-target scale silently drifted over the course of training.
# This is a very plausible explanation for the critic_loss/actor_loss
# divergence in the 200-episode log (server_0 in particular): the target
# the critic is chasing was never actually stable, even with the twin
# critics, target-policy smoothing, and Q-value clamp already in place.
#
# Fix: normalize in exactly one place. The buffer now stores the RAW env
# reward, and MADDPGCoordinator.update() (unchanged in this respect) is the
# sole place normalization happens, applied right before the reward enters
# the TD target -- which is also the only place it actually needs to be
# normalized.


def collect_predictor_dataset(num_slots):
    print(f"[Predictor] collecting {num_slots} slots of load data for pretraining")
    env = MultiAgentEdgeEnv(vrnn_encoder=None, training= True)
    columns = ["date","low_cpu","low_mem","low_num","low_ttl","mid_cpu","mid_mem","mid_num","mid_ttl","high_cpu","high_mem","high_num","high_ttl"]
    dataset = [pd.DataFrame(columns=columns) for _ in range(len(env.stations))]
    interval = 15 #minutes

    start_date_time = datetime.now()
    for slot in range(num_slots):
        for s in env.stations:
            row = [start_date_time.strftime("%Y-%m-%d %H:%M:%S")]
            for t in generate_slot_arrivals(s.id, slot):
                s.add_task(t)
            row.extend(s.local_load_vector())
            dataset[s.id].loc[len(dataset[s.id])] = row
        start_date_time += timedelta(minutes=interval)
    
    for s in env.stations:
        dataset[s.id].to_csv(cfg.LOAD_DIR / f"[{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}]raw_load_for_bs_{s.id}.csv", index= False)
    print("Saved all the dataset")
    
     
            
        

def collect_vrnn_dataset(num_slots=10080):
    """Run a VRNN-free env forward with random traffic to harvest load
    sequences for VRNN pretraining. CONFUSION: ideally the VRNN should be
    pretrained on load distributions representative of the FINAL trained
    policy's behaviour, but that's circular (policy doesn't exist yet). Using
    random/heuristic traffic as a bootstrap, matching what you described
    ("train the VRNN by the generated aggregated load" before MADDPG).

    Returns (history_dataset, next_dataset) matching train_vrnn's expected
    shapes: history_dataset is (N, VRNN_SEQ_LEN, D) -- 16 known steps -- and
    next_dataset is (N, D) -- the TRUE 17th step immediately following each
    16-step window. These come from one continuous slid window of length
    VRNN_SEQ_LEN + 1 so the label is always the real next slot, not just
    another window start.
    """

    print(f"[VRNN] collecting {num_slots} slots of load data for pretraining")
    env = MultiAgentEdgeEnv(vrnn_encoder=None, training= True)
    dataset = []
    for slot in range(num_slots):
        for s in env.stations:
            for t in generate_slot_arrivals(s.id, slot):
                s.add_task(t)
        loads = []
        for s in env.stations:
            loads.extend(s.local_load_vector())
        dataset.append(np.array(loads, dtype=np.float32))

    window_len = cfg.VRNN_SEQ_LEN + 1  # 16 history steps + 1 true next-step label
    if len(dataset) < window_len:
        raise ValueError(
            f"[VRNN] collected only {len(dataset)} slots, need at least "
            f"{window_len} (VRNN_SEQ_LEN + 1) to form a single training sample."
        )

    windows = np.array(
        [dataset[i:i + window_len] for i in range(len(dataset) - window_len + 1)],
        dtype=np.float32,
    )  # (N, 17, D)

    history_dataset = windows[:, :cfg.VRNN_SEQ_LEN, :]  # (N, 16, D)
    next_dataset = windows[:, cfg.VRNN_SEQ_LEN, :]       # (N, D) -- true 17th step

    print(f"[VRNN] collected {len(history_dataset)} samples: "
          f"history {history_dataset.shape}, next {next_dataset.shape}")
    return history_dataset, next_dataset


def run_training(episodes=500, slot_per_episode=10080,
                  vrnn_checkpoint="vrnn.pt", pretrain_vrnn=False, device=cfg.DEVICE):
    if pretrain_vrnn:
        dataset, target = collect_vrnn_dataset(num_slots=cfg.ROLLOUT_LEN)
        print(f"Collected VRNN dataset of shape: {np.array(dataset).shape}")
        train_vrnn(dataset,target, epochs=100, save_path=vrnn_checkpoint)
        return

if __name__ == "__main__":

    print(f"Runnung {cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]} on topology TOPO[{cfg.CURRENT_TOPOLOGY}]")
    collect_predictor_dataset(cfg.PREDICTOR_DATA_SLOT)
    # print(f"Runnung {cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]} on topology TOPO[{cfg.CURRENT_TOPOLOGY}]")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--episodes", type=int, default=cfg.NUM_ITERATIONS)
    # parser.add_argument("--slots", type=int, default=cfg.ROLLOUT_LEN)
    # parser.add_argument("--pretrain_vrnn", action="store_true")
    # parser.add_argument("--vrnn_checkpoint", type=str, default=f"{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}_vrnn.pt")
    # args = parser.parse_args()

    # run_training(
    #     episodes=args.episodes,
    #     slot_per_episode=args.slots,
    #     vrnn_checkpoint=args.vrnn_checkpoint,
    #     pretrain_vrnn=args.pretrain_vrnn,
    # )