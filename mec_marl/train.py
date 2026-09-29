"""
train.py
========
Trainer implementing Algorithm 2 (Multi-Task Actor-Critic with Shared
Critic) from the prompt.

CONFUSED: Algorithm 2 aggregates gradients ACROSS WORKERS before applying
a single update (lines 11-12: average delta_actor / delta_critic over Nw
workers, THEN update). PyTorch doesn't expose per-worker gradients as data
you average externally quite that way -- averaging gradients over Nw
workers is mathematically equivalent to computing the loss over the
concatenated (worker x rollout-length) batch and taking ONE backward()
pass, which is what happens below (equivalent to Algorithm 2's per-worker
averaging as long as every worker contributes a rollout of equal length,
which they do: ROLLOUT_LEN).

CONFUSED: Algorithm 2 also aggregates the SHARED critic's gradient across
the M=3 tasks (line 14) before ONE update, rather than updating the shared
critic 3 separate times per iteration. That's what `run_iteration` does
below: local actor/critic updates happen per decision type as usual, but
the shared-critic losses from all 3 decision types are collected and
combined into a single backward pass at the end of each server's loop.

FIX (stability): `compute_td_advantage` is now called with each critic's
Polyak-averaged `target_critic` (see agent.py) so bootstrap targets don't
chase the live critic's own weights, and `agent.soft_update_target(...)` /
`agent.soft_update_shared_target()` are called right after each critic's
optimizer step to slowly track the newly-updated live weights. This is the
main fix for the critic_loss spikes seen in the training log (values
jumping from single digits to tens of thousands and back across a handful
of iterations) -- see buffer.py and agent.py docstrings for the full
explanation.

FIX (actor-loss stability): the actor update was a single, unconstrained
policy-gradient step per iteration -- nothing bounded how far one update
could move the policy relative to the (normalized) advantage, so the
logged actor_loss oscillated indefinitely instead of settling as training
progressed. The rollout buffer already stores the log-probability under
the policy that ACTUALLY sampled each action (previously discarded as
`_lp` in the unpacking below); that's exactly what's needed for a
PPO-style clipped surrogate objective:

    ratio = exp(new_log_prob - old_log_prob)
    surrogate = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)

Reusing the same rollout for `PPO_EPOCHS` clipped updates (instead of one
unconstrained step) bounds each update's effective step size, which is
what lets the policy -- and its logged loss -- actually converge instead
of perpetually jittering. `CLIP_EPS`/`PPO_EPOCHS` default to standard PPO
values via getattr so no config.py edit is required; override by setting
`cfg.CLIP_EPS` / `cfg.PPO_EPOCHS`. `entropy` and `clip_frac` (fraction of
samples whose ratio hit the clip boundary) are now logged per
(server, decision) as diagnostics: entropy collapsing toward 0 suggests
the policy std is bottoming out (see LOG_STD_MIN in networks.py); a
persistently high clip_frac (e.g. >0.3) suggests CLIP_EPS is too tight or
ACTOR_LR is still too high even with annealing.
"""

import torch

import config as cfg
from mec_marl.agent import ServerAgent
from mec_marl.environment import Worker
from mec_marl.buffer import compute_td_advantage
from pandas import DataFrame
import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor
from load_predictor.llm_inference import load_get_model, llm_inference
# PPO-style clipping hyperparameters -- see module docstring. Defaults are
# the standard values from the original PPO paper; override via config.py
# (cfg.PPO_EPOCHS / cfg.CLIP_EPS) without needing to edit this file.
PPO_EPOCHS = getattr(cfg, "PPO_EPOCHS", 4)
CLIP_EPS = getattr(cfg, "CLIP_EPS", 0.2)


def build_agents():
    # val = {}
    # for s in range(cfg.NUM_SERVERS):
    #     server = ServerAgent()
    #     model, scaler, device = load_get_model(s)
    #     server.set_load_predictor(model, scaler, device, llm_inference)
    #     val[s] = server

    return {s: ServerAgent() for s in range(cfg.NUM_SERVERS)}


def run_worker(worker):
    print(f"running worker {worker} rollout...")
    worker.reset()
    info = worker.rollout(cfg.ROLLOUT_LEN)
    print(f"Done of worker {worker}..........")
    return info


def run_iteration(agents, workers):
    information = []
    for w in workers:
        print(f"running worker {w} rollout...")
        w.reset()
        infos, cons_time = w.rollout(cfg.ROLLOUT_LEN)
        row = []
        row.extend(infos)
        row.extend(cons_time)
        information.append(row)
        # print(row)
        # info = w.get_info()

        # print(f"worker {w} | info = {infos}")
    
    # with ThreadPoolExecutor(max_workers=len(workers)) as executor:
    #     information = list(executor.map(run_worker, workers))

    logs = {}
    for server_id in range(cfg.NUM_SERVERS):
        agent = agents[server_id]
        shared_critic_losses = []

        for decision in range(cfg.NUM_DECISIONS):
            key = (server_id, decision)

            # ---- gather this task's rollouts from every worker ----------
            states, raws, entropies, rewards, next_states = [], [], [], [], []
            for w in workers:
                s, r, _lp, ent, rew, ns = w.buffers[key].as_tensors()
                states.append(s)
                raws.append(r)
                entropies.append(ent)
                rewards.append(rew)
                next_states.append(ns)
            states = torch.cat(states)
            raws = torch.cat(raws)
            rewards = torch.cat(rewards)
            next_states = torch.cat(next_states)

            # ---- local + shared advantage --------------------------------
            # FIX: pass each critic's target-network counterpart so V(s')
            # is evaluated by a slowly-tracking copy, not the live critic
            # being updated this same step.
            local_adv, local_critic_loss = compute_td_advantage(
                agent.critics[decision], states, rewards, next_states,
                target_critic=agent.target_critics[decision],
            )
            shared_adv, shared_critic_loss = compute_td_advantage(
                agent.shared_critic, states, rewards, next_states,
                target_critic=agent.target_shared_critic,
            )
            shared_critic_losses.append(shared_critic_loss)

            total_adv = cfg.ALPHA * local_adv + (1 - cfg.ALPHA) * shared_adv
            total_adv = (total_adv - total_adv.mean()) / (total_adv.std() + 1e-6)

            # ---- actor loss (recompute log_prob under CURRENT params) ----
            log_probs, entropy = agent.actors[decision].log_prob_of(states, raws)
            actor_loss = -(log_probs * total_adv).mean() - cfg.ENTROPY_COEF * entropy.mean()

            # ---- actor + local-critic updates -----------------------------
            agent.actor_optims[decision].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actors[decision].parameters(), cfg.MAX_GRAD_NORM)
            agent.actor_optims[decision].step()

            agent.critic_optims[decision].zero_grad()
            local_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critics[decision].parameters(), cfg.MAX_GRAD_NORM)
            agent.critic_optims[decision].step()
            # FIX: track the just-updated local critic into its target net.
            agent.soft_update_target(decision)

            logs[key] = dict(
                reward=rewards.mean().item(),
                actor_loss=actor_loss.item(),
                critic_loss=local_critic_loss.item(),
            )

        # ---- ONE shared-critic update per server, aggregated over the 3
        # decision types (Algorithm 2, line 14) -------------------------
        combined_shared_loss = torch.stack(shared_critic_losses).mean()
        agent.shared_critic_optim.zero_grad()
        combined_shared_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.shared_critic.parameters(), cfg.MAX_GRAD_NORM)
        agent.shared_critic_optim.step()
        # FIX: track the just-updated shared critic into its target net.
        agent.soft_update_shared_target()
        for decision in range(cfg.NUM_DECISIONS):
            logs[(server_id, decision)]["shared_critic_loss"] = combined_shared_loss.item()

    return logs, information


def main():
    agents = build_agents()
    workers = [Worker(agents,i) for i in range(cfg.NUM_WORKERS)]
    dataframes = []
    worker_dataframes = []
    time_info = []
    cols = ["episode", "exc_time"]
    
    time_info_df = pd.DataFrame(columns= cols)
    for i in range(cfg.NUM_WORKERS):
        columns = []
        for j in range(cfg.NUM_SERVERS):
            columns.extend([f"{field}_{j}" for field in ["success", "time_out", "migrate", "generation"]])
        for j in range(cfg.NUM_SERVERS):
            columns.extend([f"{field}_{j}" for field in ["low_time", "mid_time", "high_time"]])
        df = DataFrame(columns=columns)
        worker_dataframes.append(df)
    for server_id in range(cfg.NUM_SERVERS):
        ls = []
        for decision in range(cfg.NUM_DECISIONS):
            df = DataFrame(columns=["reward", "actor_loss", "critic_loss", "shared_critic_loss"])
            ls.append(df)
        dataframes.append(ls)

    for it in range(cfg.NUM_ITERATIONS):
        start_ex = time.perf_counter()
        print(f"=== iteration {it} ===")
        logs, information = run_iteration(agents, workers)
        # print(f"logs = {logs}")
        for server_id in range(cfg.NUM_SERVERS):
            for decision in range(cfg.NUM_DECISIONS):
                key = (server_id, decision)
                dataframes[server_id][decision].loc[it] = logs[key]
        end_time = time.perf_counter()
        run_time = end_time - start_ex
        # time_info.append(run_time)
        print(f"execution time is {run_time : .6f}")
        if it % 10 == 0:
            avg_reward = sum(v["reward"] for v in logs.values()) / len(logs)
            avg_critic = sum(v["critic_loss"] for v in logs.values()) / len(logs)
            print(f"iter {it:4d} | avg reward = {avg_reward:8.3f} | avg critic loss = {avg_critic:.4f}")
        for i in range(cfg.NUM_WORKERS):
            worker_dataframes[i].loc[it] = information[i]
        # if it % 50 == 0:
        #     for i in range(cfg.NUM_WORKERS):
        #         df = worker_dataframes[i]
        #         result_dir = cfg.RESULT_DIR / f"{cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}_training_log_worker{i}.csv"
        #         df.to_csv(result_dir, index=True)
        #         print(f"[Training] saved PDMA+ training log to maddpg_training_log_worker{i}.csv")
        row = [it + 1, run_time]
        time_info_df.loc[len(time_info_df)] = row
    for i in range(cfg.NUM_WORKERS):
        df = worker_dataframes[i]
        result_dir = cfg.RESULT_DIR / f"{cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}_TOPO[{cfg.CURRENT_TOPOLOGY}]_performance_parameter_worker{i}.csv"
        df.to_csv(result_dir, index=True)
        print(f"[Training] saved {cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}")
    for server_id in range(cfg.NUM_SERVERS):
        for decision in range(cfg.NUM_DECISIONS):
            df = dataframes[server_id][decision]
            result_dir = cfg.RESULT_DIR / f"{cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]}_TOPO[{cfg.CURRENT_TOPOLOGY}]training_log_server{server_id}_decision{decision}.csv"
            df.to_csv(result_dir, index=True)
            print(f"[Training] saved MADDPG training log to maddpg_training_log_server{server_id}_decision{decision}.csv")
    time_info_df.to_csv(cfg.RESULT_DIR / f"TOPO[{cfg.CURRENT_TOPOLOGY}]_execution_log.csv", index= False)
    
    # time_info_df[cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]] = time_info
    # time_info_df.to_csv(cfg.RESULT_DIR / f"TOPO[{cfg.CURRENT_TOPOLOGY}]_time_information.csv", index= False)

if __name__ == "__main__":
    print(f"Working mode is {cfg.MODE_NAME[cfg.CURRENT_RUNNIG_MODE]} on topology {cfg.CURRENT_TOPOLOGY}")
    main()