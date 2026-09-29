"""
End-to-end training entry point.

Usage (illustrative -- CONFUSION: no CLI spec given, kept minimal):
    python -m training.train --pretrain_vrnn --episodes 1000
"""
import argparse
import numpy as np

from madrl_edge import config
from madrl_edge.core.environment import MultiAgentEdgeEnv
from madrl_edge.core.traffic import generate_slot_arrivals
from madrl_edge.agents.maddpg_agent import MADDPGCoordinator
from madrl_edge.agents.replay_buffer import MultiAgentReplayBuffer
from madrl_edge.models.new_vrnnn import FrozenVRNNEncoder, train_vrnn
from datetime import datetime, timedelta
import pandas as pd
import config as cfg
import time

# from utils import losses

# from utils import losses

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
    
     
            
        

# def collect_vrnn_dataset(num_slots=10080):
#     """Run a VRNN-free env forward with random traffic to harvest load
#     sequences for VRNN pretraining. CONFUSION: ideally the VRNN should be
#     pretrained on load distributions representative of the FINAL trained
#     policy's behaviour, but that's circular (policy doesn't exist yet). Using
#     random/heuristic traffic as a bootstrap, matching what you described
#     ("train the VRNN by the generated aggregated load" before MADDPG)."""

#     print(f"[VRNN] collecting {num_slots} slots of load data for pretraining")
#     env = MultiAgentEdgeEnv(vrnn_encoder=None)
#     dataset = []
#     for slot in range(num_slots):
#         for s in env.stations:
#             for t in generate_slot_arrivals(s.id, slot):
#                 s.add_task(t)
#         loads = []
#         for s in env.stations:
#             loads.extend(s.local_load_vector())
#         dataset.append(np.array(loads, dtype=np.float32))
#         # cheap random allocation/migration just to keep queues moving
#         dummy_actions = [np.concatenate([
#             np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3]),
#             np.zeros(6),
#         ]) for _ in env.stations]
#         env.step(dummy_actions)
#     sequences = np.array([dataset[i:i + cfg.VRNN_SEQ_LEN]
#         for i in range(len(dataset) - cfg.VRNN_SEQ_LEN + 1)])
#     print(f"[VRNN] collected {len(sequences)} sequences of shape {sequences.shape}")
#     return sequences

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
        # cheap random allocation/migration just to keep queues moving
        # dummy_actions = [np.concatenate([
        #     np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3]),
        #     np.zeros(6),
        # ]) for _ in env.stations]
        # env.step(dummy_actions)

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
    

    

    vrnn_encoder = FrozenVRNNEncoder(vrnn_checkpoint, device=device)
    env = MultiAgentEdgeEnv(vrnn_encoder=vrnn_encoder)
    coordinator = MADDPGCoordinator(cfg.NUM_SERVERS,device=device)
    buffer = MultiAgentReplayBuffer(
        cfg.BUFFER_CAPACITY, cfg.NUM_SERVERS, cfg.OBS_DIM, cfg.MADDPG_ACTION_DIM
    )
    
    global_step = 0
    columns = ["episode"]
    oth = []
    for i in range(cfg.NUM_SERVERS):
        columns.append(f"server_{i}_reward")
        columns.append(f"server_{i}_actor_loss")
        columns.append(f"server_{i}_critic_loss")
        columns.append(f"server_{i}_total_success")
        columns.append(f"server_{i}_total_violation")
        columns.append(f"server_{i}_completion_rate")
        oth.extend([f"low-{i}", f"mid-{i}", f"high-{i}"])
    columns.extend(oth)
    maddpg_training_dataframe = pd.DataFrame(columns=columns)
    columns = ["episode", "req_time"]
    execution_log_dataframe = pd.DataFrame(columns= columns)
    for ep in range(episodes):
        start = time.perf_counter()
        obs_n, masks_n = env.reset()
        ep_reward = np.zeros(cfg.NUM_SERVERS)
        ep_actor_loss = np.zeros(cfg.NUM_SERVERS)
        ep_critic_loss = np.zeros(cfg.NUM_SERVERS)
        ep_total_success = np.zeros(cfg.NUM_SERVERS)
        ep_total_violation = np.zeros(cfg.NUM_SERVERS)
        ep_completion_rate = np.zeros(cfg.NUM_SERVERS)
        # print(f"[ep {ep}] starting episode with initial observations {[o.tolist() for o in obs_n]}")
        # break
        print(f"Epoch {ep+1}/{episodes} starting...")
        cons_time = np.zeros(cfg.NUM_SERVERS * 3)
        for slot in range(slot_per_episode):
            for s in env.stations:
                for t in generate_slot_arrivals(s.id, env.slot):
                    s.add_task(t)
            obs_n, masks_n = env.observe_all()

            noise_std = max(0.3 * (1 - global_step / 50_000), 0.02)
            action_n = coordinator.act_all(obs_n, migration_masks=masks_n, noise_std=noise_std)

            next_obs_n, next_masks_n, reward_n, done_n, info_n, req_time = env.step(action_n)
            buffer.push(obs_n, action_n, reward_n, next_obs_n, done_n)
            cons_time += np.array(req_time)
            losses = coordinator.update(buffer)
            # print(f"[ep {ep} slot {slot}] losses: {losses} reward: {reward_n} completion_rate: {[info['completion_rate'] for info in info_n]}")
            ep_reward += np.array(reward_n)
            
            if losses is not None:
                
                actor_losses = [
                    v["actor_loss"] if v["actor_loss"] is not None else 0.0
                    for v in losses.values()
                ]
                critic_losses = [
                    v["critic_loss"] if v["critic_loss"] is not None else 0.0
                    for v in losses.values()
                ]
                ep_actor_loss += np.array(actor_losses)
                ep_critic_loss += np.array(critic_losses)
                # print(losses)
            else:
                ep_actor_loss += np.zeros(cfg.NUM_SERVERS)
                ep_critic_loss += np.zeros(cfg.NUM_SERVERS)
            ep_total_success += np.array([info["n_completed"] for info in info_n])
            ep_total_violation += np.array([info["n_dropped"] for info in info_n])
            ep_completion_rate += np.array([info["completion_rate"] for info in info_n])
            obs_n, masks_n = next_obs_n, next_masks_n
            global_step += 1

        row = [ep + 1]
        for i in range(cfg.NUM_SERVERS):
            row.append(ep_reward[i])
            row.append(ep_actor_loss[i])
            row.append(ep_critic_loss[i])
            row.append(ep_total_success[i])
            row.append(ep_total_violation[i])
            row.append(ep_completion_rate[i])
        row.extend(cons_time / slot_per_episode)
        maddpg_training_dataframe.loc[len(maddpg_training_dataframe)] = row
        if ep % 10 == 0:
            avg_completion = np.mean([info["completion_rate"] for info in info_n])
            print(f"[ep {ep}] mean_reward={ep_reward.mean():.3f} "
                  f"completion_rate={avg_completion:.3f}")
        end_ex = time.perf_counter()
        run_time = end_ex - start
        row = [ep + 1, run_time]
        execution_log_dataframe.loc[len(execution_log_dataframe)] = row
        print(f"Execution time is {run_time}")

    result_dir = cfg.RESULT_DIR / f"{cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_training_log.csv"
    maddpg_training_dataframe.to_csv(result_dir, index=False)
    result_dir = cfg.RESULT_DIR / f"{cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_execution_log.csv"
    execution_log_dataframe.to_csv(result_dir, index=False)
    print(f"[Training] saved MADDPG training log to maddpg_training_log.csv")
    return coordinator


if __name__ == "__main__":
    print(f"Runnung {cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]} on topology TOPO[{cfg.CURRENT_TOPOLOGY}]")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=cfg.NUM_ITERATIONS)
    parser.add_argument("--slots", type=int, default=cfg.ROLLOUT_LEN)
    parser.add_argument("--pretrain_vrnn", action="store_true")
    parser.add_argument("--vrnn_checkpoint", type=str, default=f"{cfg.TOPO_NAME[cfg.CURRENT_TOPOLOGY]}_vrnn.pt")
    args = parser.parse_args()

    # collect_predictor_dataset(1100)
    run_training(
        episodes=args.episodes,
        slot_per_episode=args.slots,
        vrnn_checkpoint=args.vrnn_checkpoint,
        pretrain_vrnn=args.pretrain_vrnn,
    )
