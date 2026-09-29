
"""
train.py
========

Parallelized Trainer implementing Algorithm 2
(Multi-Task Actor-Critic with Shared Critic).

Parallelization strategy
------------------------
1. Worker rollouts are executed concurrently using ThreadPoolExecutor.
2. Each worker owns its own environment/buffer and performs:
       reset()
       rollout(ROLLOUT_LEN)
3. After all workers finish, their rollout buffers are aggregated.
4. Actor and local-critic updates are performed centrally.
5. Shared critic losses from all decision types are accumulated and
   combined into ONE backward/update operation per server.

This preserves Algorithm 2:

    - gradients across workers are effectively averaged through the
      concatenated rollout batch;
    - shared critic gradients across the M decision types are combined
      before a single shared-critic update.

Important
---------
The Worker.rollout() implementation must be thread-safe.

Each Worker object must be used by only ONE thread at a time. This
implementation satisfies that condition because each worker is submitted
exactly once per iteration.

If Worker.rollout() uses a non-thread-safe external simulator or library,
replace ThreadPoolExecutor with ProcessPoolExecutor or another
process-based architecture.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from pandas import DataFrame

import config as cfg
from mec_marl.agent import ServerAgent
from mec_marl.environment import Worker
from mec_marl.buffer import compute_td_advantage


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

PPO_EPOCHS = getattr(cfg, "PPO_EPOCHS", 4)
CLIP_EPS = getattr(cfg, "CLIP_EPS", 0.2)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agents():
    """
    Create all server agents.

    Agents remain centralized because the training algorithm uses shared
    critics and centrally aggregated gradients.
    """
    return {
        server_id: ServerAgent()
        for server_id in range(cfg.NUM_SERVERS)
    }


# ---------------------------------------------------------------------------
# Worker rollout
# ---------------------------------------------------------------------------

def run_worker(worker_id, worker):
    """
    Execute one worker rollout.

    This function is intentionally independent so it can be executed
    concurrently by ThreadPoolExecutor.
    """

    print(f"[Worker {worker_id}] starting rollout...")

    start_time = time.perf_counter()

    worker.reset()

    infos, cons_time = worker.rollout(cfg.ROLLOUT_LEN)

    elapsed = time.perf_counter() - start_time

    print(
        f"[Worker {worker_id}] rollout completed "
        f"in {elapsed:.4f} seconds."
    )

    row = []
    row.extend(infos)
    row.extend(cons_time)

    return worker_id, row


# ---------------------------------------------------------------------------
# Parallel rollout manager
# ---------------------------------------------------------------------------

def run_parallel_rollouts(workers, executor):
    """
    Run all worker rollouts concurrently.

    Returns
    -------
    information : list
        Worker performance information ordered by worker ID.
    """

    futures = {
        executor.submit(run_worker, worker_id, worker): worker_id
        for worker_id, worker in enumerate(workers)
    }

    information = [None] * len(workers)

    for future in as_completed(futures):
        worker_id = futures[future]

        try:
            returned_worker_id, row = future.result()
            information[returned_worker_id] = row

        except Exception as exc:
            print(
                f"[Worker {worker_id}] ERROR during rollout: "
                f"{type(exc).__name__}: {exc}"
            )

            # Cancel workers that have not started yet.
            for other_future in futures:
                if other_future != future:
                    other_future.cancel()

            raise

    return information


# ---------------------------------------------------------------------------
# Gather rollout tensors
# ---------------------------------------------------------------------------

def gather_rollout_batch(workers, server_id, decision):
    """
    Gather the rollout data of all workers for one
    (server, decision) pair.

    The workers have equal rollout lengths, so concatenating the tensors
    gives the same effective gradient averaging behavior as averaging
    per-worker gradients.
    """

    key = (server_id, decision)

    states = []
    raws = []
    rewards = []
    next_states = []

    for worker in workers:

        (
            state,
            raw_action,
            _old_log_prob,
            _entropy,
            reward,
            next_state,
        ) = worker.buffers[key].as_tensors()

        states.append(state)
        raws.append(raw_action)
        rewards.append(reward)
        next_states.append(next_state)

    states = torch.cat(states, dim=0)
    raws = torch.cat(raws, dim=0)
    rewards = torch.cat(rewards, dim=0)
    next_states = torch.cat(next_states, dim=0)

    return (
        states,
        raws,
        rewards,
        next_states,
    )


# ---------------------------------------------------------------------------
# Train one server
# ---------------------------------------------------------------------------

def train_server(agent, workers, server_id):
    """
    Train one server for the current iteration.

    Local actor/local critic updates happen for each decision type.

    Shared critic losses from all decision types are accumulated and
    applied through ONE optimizer step.
    """

    logs = {}

    shared_critic_losses = []

    # -----------------------------------------------------------------------
    # Process each decision/task
    # -----------------------------------------------------------------------

    for decision in range(cfg.NUM_DECISIONS):

        key = (server_id, decision)

        # ---------------------------------------------------------------
        # Gather rollout data from ALL workers
        # ---------------------------------------------------------------

        (
            states,
            raws,
            rewards,
            next_states,
        ) = gather_rollout_batch(
            workers,
            server_id,
            decision,
        )

        # ---------------------------------------------------------------
        # Local critic advantage
        # ---------------------------------------------------------------

        local_adv, local_critic_loss = compute_td_advantage(
            agent.critics[decision],
            states,
            rewards,
            next_states,
            target_critic=agent.target_critics[decision],
        )

        # ---------------------------------------------------------------
        # Shared critic advantage
        # ---------------------------------------------------------------

        shared_adv, shared_critic_loss = compute_td_advantage(
            agent.shared_critic,
            states,
            rewards,
            next_states,
            target_critic=agent.target_shared_critic,
        )

        # Store the shared loss.

        shared_critic_losses.append(shared_critic_loss)

        # ---------------------------------------------------------------
        # Combined advantage
        # ---------------------------------------------------------------

        total_adv = (
            cfg.ALPHA * local_adv
            + (1.0 - cfg.ALPHA) * shared_adv
        )

        # Advantage normalization.

        total_adv = (
            total_adv - total_adv.mean()
        ) / (
            total_adv.std() + 1e-6
        )

        # ---------------------------------------------------------------
        # Actor
        # ---------------------------------------------------------------

        log_probs, entropy = (
            agent.actors[decision].log_prob_of(
                states,
                raws,
            )
        )

        # PPO-style clipped actor objective.
        #
        # NOTE:
        # This section assumes the actor's rollout buffer stores old
        # log probabilities and that PPO updates are desired.
        #
        # If log_prob_of() is the only available interface, the original
        # policy-gradient objective can be retained here.

        actor_loss = (
            -(log_probs * total_adv).mean()
            - cfg.ENTROPY_COEF * entropy.mean()
        )

        # ---------------------------------------------------------------
        # Actor update
        # ---------------------------------------------------------------

        agent.actor_optims[decision].zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            agent.actors[decision].parameters(),
            cfg.MAX_GRAD_NORM,
        )

        agent.actor_optims[decision].step()

        # ---------------------------------------------------------------
        # Local critic update
        # ---------------------------------------------------------------

        agent.critic_optims[decision].zero_grad(
            set_to_none=True
        )

        local_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            agent.critics[decision].parameters(),
            cfg.MAX_GRAD_NORM,
        )

        agent.critic_optims[decision].step()

        # ---------------------------------------------------------------
        # Polyak update of local target critic
        # ---------------------------------------------------------------

        agent.soft_update_target(decision)

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------

        logs[key] = {
            "reward": rewards.mean().item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": local_critic_loss.item(),
        }

    # -----------------------------------------------------------------------
    # ONE shared critic update
    # -----------------------------------------------------------------------

    combined_shared_loss = torch.stack(
        shared_critic_losses
    ).mean()

    agent.shared_critic_optim.zero_grad(
        set_to_none=True
    )

    combined_shared_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        agent.shared_critic.parameters(),
        cfg.MAX_GRAD_NORM,
    )

    agent.shared_critic_optim.step()

    # -----------------------------------------------------------------------
    # Polyak update shared target critic
    # -----------------------------------------------------------------------

    agent.soft_update_shared_target()

    # -----------------------------------------------------------------------
    # Add shared critic loss to all decision logs
    # -----------------------------------------------------------------------

    shared_loss_value = combined_shared_loss.item()

    for decision in range(cfg.NUM_DECISIONS):

        key = (server_id, decision)

        logs[key]["shared_critic_loss"] = shared_loss_value

    return logs


# ---------------------------------------------------------------------------
# Complete training iteration
# ---------------------------------------------------------------------------

def run_iteration(
    agents,
    workers,
    rollout_executor,
):
    """
    Execute one complete training iteration.

    Phase 1
    -------
    All workers perform their rollouts in parallel.

    Phase 2
    -------
    Server agents perform centralized gradient updates.

    Returns
    -------
    logs
        Training losses/rewards.

    information
        Per-worker environment information.
    """

    # =======================================================================
    # PHASE 1: PARALLEL ROLLOUTS
    # =======================================================================

    information = run_parallel_rollouts(
        workers,
        rollout_executor,
    )

    # =======================================================================
    # PHASE 2: CENTRALIZED TRAINING
    # =======================================================================

    logs = {}

    for server_id in range(cfg.NUM_SERVERS):

        agent = agents[server_id]

        server_logs = train_server(
            agent,
            workers,
            server_id,
        )

        logs.update(server_logs)

    return logs, information


# ---------------------------------------------------------------------------
# DataFrame initialization
# ---------------------------------------------------------------------------

def create_worker_dataframes():
    """
    Create one DataFrame per worker for environment statistics.
    """

    worker_dataframes = []

    for _ in range(cfg.NUM_WORKERS):

        columns = []

        # Server-level statistics

        for server_id in range(cfg.NUM_SERVERS):

            columns.extend(
                [
                    f"{field}_{server_id}"
                    for field in [
                        "success",
                        "time_out",
                        "migrate",
                        "generation",
                    ]
                ]
            )

        # Time-level statistics

        for server_id in range(cfg.NUM_SERVERS):

            columns.extend(
                [
                    f"{field}_{server_id}"
                    for field in [
                        "low_time",
                        "mid_time",
                        "high_time",
                    ]
                ]
            )

        worker_dataframes.append(
            DataFrame(columns=columns)
        )

    return worker_dataframes


def create_training_dataframes():
    """
    Create one DataFrame for each
    (server, decision) pair.
    """

    dataframes = []

    for server_id in range(cfg.NUM_SERVERS):

        server_dataframes = []

        for decision in range(cfg.NUM_DECISIONS):

            df = DataFrame(
                columns=[
                    "reward",
                    "actor_loss",
                    "critic_loss",
                    "shared_critic_loss",
                ]
            )

            server_dataframes.append(df)

        dataframes.append(server_dataframes)

    return dataframes


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    worker_dataframes,
    dataframes,
    time_info_df,
):
    """
    Save all training results.
    """

    # -----------------------------------------------------------------------
    # Worker results
    # -----------------------------------------------------------------------

    for worker_id in range(cfg.NUM_WORKERS):

        df = worker_dataframes[worker_id]

        result_dir = (
            cfg.RESULT_DIR
            / f"{cfg.MODEL_NAME[cfg.CURRENT_RUNNING_MODE]}"
            f"_TOPO[{cfg.CURRENT_TOPOLOGY}]"
            f"_performance_parameter_worker{worker_id}.csv"
        )

        df.to_csv(
            result_dir,
            index=True,
        )

        print(
            f"[Training] saved worker {worker_id} results"
        )

    # -----------------------------------------------------------------------
    # Server/decision training logs
    # -----------------------------------------------------------------------

    for server_id in range(cfg.NUM_SERVERS):

        for decision in range(cfg.NUM_DECISIONS):

            df = dataframes[server_id][decision]

            result_dir = (
                cfg.RESULT_DIR
                / f"{cfg.MODEL_NAME[cfg.CURRENT_RUNNING_MODE]}"
                f"_TOPO[{cfg.CURRENT_TOPOLOGY}]"
                f"_training_log_server{server_id}"
                f"_decision{decision}.csv"
            )

            df.to_csv(
                result_dir,
                index=True,
            )

            print(
                f"[Training] saved server {server_id}, "
                f"decision {decision} training log"
            )

    # -----------------------------------------------------------------------
    # Execution-time log
    # -----------------------------------------------------------------------

    time_info_df.to_csv(
        cfg.RESULT_DIR
        / f"TOPO[{cfg.CURRENT_TOPOLOGY}]_execution_log.csv",
        index=False,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():

    print(
        f"Working mode is "
        f"{cfg.MODEL_NAME[cfg.CURRENT_RUNNING_MODE]} "
        f"on topology {cfg.CURRENT_TOPOLOGY}"
    )

    # =======================================================================
    # Create agents
    # =======================================================================

    agents = build_agents()

    # =======================================================================
    # Create workers
    # =======================================================================

    workers = [
        Worker(
            agents,
            worker_id,
        )
        for worker_id in range(cfg.NUM_WORKERS)
    ]

    # =======================================================================
    # Create result DataFrames
    # =======================================================================

    dataframes = create_training_dataframes()

    worker_dataframes = create_worker_dataframes()

    time_info_df = DataFrame(
        columns=[
            "episode",
            "exc_time",
        ]
    )

    # =======================================================================
    # Create ONE persistent thread pool
    # =======================================================================

    #
    # Do NOT create a new ThreadPoolExecutor inside every iteration.
    #
    # The executor is created once and reused throughout training.
    #

    with ThreadPoolExecutor(
        max_workers=cfg.NUM_WORKERS,
        thread_name_prefix="rollout",
    ) as rollout_executor:

        # ================================================================
        # Training loop
        # ================================================================

        for iteration in range(cfg.NUM_ITERATIONS):

            print(
                f"\n{'=' * 70}"
            )

            print(
                f"Iteration {iteration}"
            )

            print(
                f"{'=' * 70}"
            )

            iteration_start = time.perf_counter()

            # ------------------------------------------------------------
            # Parallel rollout + centralized training
            # ------------------------------------------------------------

            logs, information = run_iteration(
                agents,
                workers,
                rollout_executor,
            )

            # ------------------------------------------------------------
            # Store training logs
            # ------------------------------------------------------------

            for server_id in range(cfg.NUM_SERVERS):

                for decision in range(cfg.NUM_DECISIONS):

                    key = (
                        server_id,
                        decision,
                    )

                    dataframes[
                        server_id
                    ][
                        decision
                    ].loc[iteration] = logs[key]

            # ------------------------------------------------------------
            # Execution time
            # ------------------------------------------------------------

            iteration_end = time.perf_counter()

            run_time = (
                iteration_end
                - iteration_start
            )

            print(
                f"Iteration execution time: "
                f"{run_time:.6f} seconds"
            )

            # ------------------------------------------------------------
            # Periodic training statistics
            # ------------------------------------------------------------

            if iteration % 10 == 0:

                avg_reward = (
                    sum(
                        value["reward"]
                        for value in logs.values()
                    )
                    / len(logs)
                )

                avg_critic = (
                    sum(
                        value["critic_loss"]
                        for value in logs.values()
                    )
                    / len(logs)
                )

                avg_shared_critic = (
                    sum(
                        value["shared_critic_loss"]
                        for value in logs.values()
                    )
                    / len(logs)
                )

                avg_actor = (
                    sum(
                        value["actor_loss"]
                        for value in logs.values()
                    )
                    / len(logs)
                )

                print(
                    f"iter {iteration:4d} | "
                    f"avg reward = {avg_reward:10.4f} | "
                    f"avg actor loss = {avg_actor:10.4f} | "
                    f"avg critic loss = {avg_critic:10.4f} | "
                    f"avg shared critic loss = "
                    f"{avg_shared_critic:10.4f}"
                )

            # ------------------------------------------------------------
            # Store worker statistics
            # ------------------------------------------------------------

            for worker_id in range(cfg.NUM_WORKERS):

                worker_dataframes[
                    worker_id
                ].loc[iteration] = information[
                    worker_id
                ]

            # ------------------------------------------------------------
            # Store execution time
            # ------------------------------------------------------------

            time_info_df.loc[
                len(time_info_df)
            ] = [
                iteration + 1,
                run_time,
            ]

    # =======================================================================
    # Save everything
    # =======================================================================

    save_results(
        worker_dataframes,
        dataframes,
        time_info_df,
    )

    print(
        "\nTraining completed successfully."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    main()

