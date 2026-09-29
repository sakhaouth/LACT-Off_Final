"""
edge_server.py
===============
Pure simulation logic for one edge server: task queues, load computation,
neighbor-load combination, resource allocation/migration/sharing execution,
and the residual-time reward. No RL code lives here -- decisions come in as
plain numpy arrays produced elsewhere (see agent.py / environment.py). This
separation keeps the physics testable independently of the learning code.

FIX (stability): `compute_reward_components()` reads from
`self._completed_this_slot` / `self._dropped_this_slot`, but these lists
were previously only cleared in `reset()` (once per rollout), not per
slot. That meant the reward computed at slot t was based on EVERY task
completed/dropped since the rollout began, not just this slot's tasks --
so the reward signal grew roughly linearly with `slot` over a
ROLLOUT_LEN=500 rollout. This made the reward (and hence the TD targets
and advantages) badly non-stationary within a single rollout: the same
action produced wildly different reward magnitudes depending on when in
the rollout it happened, so the critic could never converge and the actor
was pushed around by scale drift rather than real signal. `advance_slot()`
now clears `_completed_this_slot`, `_dropped_this_slot`, and
`migrate_tasks_slot` every slot, matching `self.stats`' existing per-slot
reset behavior.
"""

from typing import List

import numpy as np

import config as cfg
from mec_marl.task import Task, LOW, MID, HIGH, TaskQueue
import pandas as pd
from collections import deque
from load_predictor.load_predictor_interface import Predictor

class EdgeServer:
    def __init__(self, server_id):
        self.id = server_id
        self.queues = {q: TaskQueue(q) for q in (LOW, MID, HIGH)}
        self.slot = 0
        self.predictor = Predictor(server_id)
        # FIX (cross-platform paths): was a hardcoded POSIX-style relative
        # string ("./csv/tasks-{id}.csv"), which assumes both a forward-slash
        # separator and that the process is launched from the project root.
        # `cfg.DATA_DIR` is a `pathlib.Path` resolved relative to config.py's
        # own location (see config.py), so this works identically on Ubuntu
        # and Windows and regardless of the current working directory.
        csv_path = cfg.TASK_DIR / f"tasks-{self.id}.csv"
        self.load_dataframes = pd.read_csv(csv_path, index_col=0)
        self.load_dataframes.index = range(len(self.load_dataframes))
        # populated fresh each slot
        self.local_load = None            # np.ndarray[12], this slot's own load
        self.neighbor_loads = {}          # {neighbor_id: np.ndarray[12]}
        self.combined_load = None         # np.ndarray[12] after weighted aggregation
        self.sharing_received = {}        # {neighbor_id: (offered_cpu, offered_mem)}
        self.migration_likelihood = {}    # {neighbor_id: probability}

        # per-slot stats, read by the trainer for logging
        self.stats = dict(success=0, timeout=0, migrated=0, generated=0)
        self.success = [0, 0, 0]
        # self.avg_time = [0.0, 0.0, 0.0]
        # self.task_count = [0, 0 ,0]
        self._completed_this_slot: List[Task] = []
        self._dropped_this_slot: List[Task] = []
        self.migrate_tasks_slot: List[Task] = []
        # self.history = deque(
        #     [[0.0] * cfg.LOCAL_STATE_DIM for _ in range(cfg.SEQ_LEN)],
        #     maxlen=cfg.SEQ_LEN
        # )
    # -- reset ----------------------------------------------------------
    
    def reset(self):
        self.queues = {q: TaskQueue(q) for q in (LOW, MID, HIGH)}
        self.slot = 0
        self.local_load = None
        self.neighbor_loads = {}
        self.combined_load = None
        self.sharing_received = {}
        self.migration_likelihood = {}
        self.stats = dict(success=0, timeout=0, migrated=0, generated=0)
        self._completed_this_slot = []
        self._dropped_this_slot = []
        self.migrate_tasks_slot = []
        self.success = [0, 0, 0]
        self.avg_time = [0.0, 0.0, 0.0]
        self.task_count = [0, 0, 0]
        self.predictor.reset()

    # -- 1. local load generation -----------------------------------------
    # def server_load_vector(queues) -> List[float]:
    #     """queues: dict priority -> TaskQueue, or list ordered [HIGH, MID, LOW]."""
    #     if isinstance(queues, dict):
    #         ordered = [queues[p] for p in cfg.PRIORITIES]
    #     else:
    #         ordered = queues
    #     vec = []
    #     for q in ordered:
    #         vec.extend(q.normalized_load())
    #     assert len(vec) == cfg.LOCAL_LOAD_DIM
    #     return vec
    def local_load_generation(self):
        """Sample new task arrivals for each queue, then compute this
        server's own load vector."""
        n_new = 0
        ind = ["LOW", "MID", "HIGH"]
        for q in (LOW, MID, HIGH):
            n_arrivals = self.load_dataframes.loc[self.slot, ind[q].lower()]
            # print(n_arrivals)
            for _ in range(n_arrivals):
                self.queues[q].add(Task.random_task(q, self.slot, cfg, self.id))
            n_new += n_arrivals
        self.stats["generated"] = n_new
        raw_load = self._compute_load_vector()
        raw_list = raw_load.tolist()
        # print("here is the raw list {r}")
        # self.history.append(raw_list)
        # predicted_load = self.inference_fun(self.history, self.model, self.scaler, self.device)
        predicted_load = self.predictor.predict(raw_list)
        self.local_load = None
        if cfg.CURRENT_RUNNIG_MODE == cfg.MINUS_PDMA or cfg.CURRENT_RUNNIG_MODE == cfg.MINUS_LACT_Off:
            self.local_load = raw_load
        else:
            self.local_load = predicted_load
        # print("pred", predicted_load)

        # print(f"server {self.id} | slot {self.slot} | generating new tasks... {self.stats} ")

    def _compute_load_vector(self):
        vec = []
        for q in (LOW, MID, HIGH):
            vec.extend(self.queues[q].compute_load())
        return np.asarray(vec, dtype=np.float32)  # len 12 (NUM_QUEUES * 4)

    # -- 2. local load sharing --------------------------------------------
    def local_load_sharing(self, state_mailboxes):
        """Publish this server's local_load into every neighbor's mailbox."""
        # print(cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id])
        for n in cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]:
            # print(n, self.id, state_mailboxes)
            state_mailboxes[n][self.id] = self.local_load

    # -- 3. load aggregation ----------------------------------------------
    def load_aggregation(self, state_mailboxes):
        """
        Combine neighbor loads, weighted by connection speed and the
        neighbor's resource capacity.

        CONFUSED: the prompt says loads are "combined ... weighted on the
        connection speed and the resource capacity" but doesn't give the
        exact formula. Implemented as a normalized weighted average:
            w_n = speed(self, n) * capacity(n)
            combined = sum_n(w_n * load_n) / sum_n(w_n)
        i.e. a neighbor with more spare capacity AND a faster link to us
        counts for more in the combined picture. If capacity should instead
        *discount* the weight (a busier neighbor's load matters less
        because it can't help anyway), invert `capacity` below.
        """
        self.neighbor_loads = dict(state_mailboxes.get(self.id, {}))
        # print(state_mailboxes)
        if not self.neighbor_loads:
            self.combined_load = np.zeros_like(self.local_load)
            return

        weights, loads = [], []
        for n, load in self.neighbor_loads.items():
            speed = cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][self.id][n]
            # ASSUMPTION: identical capacity for every server currently, so
            # this term is constant -- kept explicit so a per-server
            # capacity array can be dropped in later without touching the
            # weighting formula itself.
            capacity = cfg.SERVERS[n]["cpu_hz"] * cfg.CPU_IMPORTANCE + cfg.SERVERS[n]["memory_mb"] * cfg.MEM_IMPORTANCE
            # capacity = cfg.DEV_CAPACITY[n][0] + cfg.DEV_CAPACITY[n][1]
            weights.append(speed * capacity)
            loads.append(load)
        weights = np.asarray(weights, dtype=np.float32)
        loads = np.stack(loads, axis=0)
        weights = weights / (weights.sum() + 1e-8)
        self.combined_load = (weights[:, None] * loads).sum(axis=0)

    def get_info(self):
        # avg_time = [self.avg_time[LOW] / (self.task_count[LOW] + 1e-12), self.avg_time[MID] / (self.task_count[MID] + 1e-12), self.avg_time[HIGH] / (self.task_count[HIGH] + 1e-12)]
        # print("get info")
        # print(self.stats, avg_time)
        return self.stats, self.avg_time

    def get_state(self):
        """State fed to all 3 actors/critics: local load ++ combined neighbor load."""
        combined = self.combined_load if self.combined_load is not None else np.zeros_like(self.local_load)
        return np.concatenate([self.local_load, combined]).astype(np.float32)

    # -- 4. resource sharing decision exchange -----------------------------

    def publish_reward(self, rewards, reward_mailboxes):
        # print(rewards)
        for n in cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]:
            # print("nnnnnnnnn ", n , self.id)
            reward_mailboxes[n][self.id] = rewards[n]

    def publish_sharing_decision(self, sharing_action, resource_mailboxes):
        """
        sharing_action: 6-vector [cpu_share_low, cpu_share_mid, cpu_share_high,
                                   mem_share_low, mem_share_mid, mem_share_high],
        each in [0, 1] -- an independent fraction of that queue's OWN demand
        this server is willing to lend out to neighbors.
        """
        cpu_share, mem_share = sharing_action[:3], sharing_action[3:]
        # local_load layout per queue is [cpu, mem, tolerance, count] repeated
        # 3 times -> cpu indices 0,4,8 and mem indices 1,5,9.
        offered_cpu = float((cpu_share * self.local_load[[0, 4, 8]]).sum())
        offered_mem = float((mem_share * self.local_load[[1, 5, 9]]).sum())
        for n in cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]:
            resource_mailboxes[n][self.id] = (offered_cpu, offered_mem)

    def collect_sharing_offers(self, resource_mailboxes):
        self.sharing_received = dict(resource_mailboxes.get(self.id, {}))

    # -- 5. migration likelihood --------------------------------------------
    def compute_migration_likelihood(self):
        """
        Turn neighbors' offered (cpu, mem) capacity into a softmax
        probability of migrating a given task to each neighbor.

        CONFUSED: "calculate the migration likelihood based on the shared
        decision" isn't spelled out numerically. Implemented as a softmax
        over each neighbor's *total offered resource* (cpu + mem), so
        neighbors who offered more spare capacity are more likely
        migration targets. If likelihood should also depend on connection
        speed / distance (closer/faster neighbors preferred even if they
        offer less), fold `cfg.CONNECTION_SPEED[self.id, n]` into the score
        below.
        """
        neighbors = list(self.sharing_received.keys())
        if not neighbors:
            self.migration_likelihood = {}
            return
        scores = np.array([sum(self.sharing_received[n]) for n in neighbors], dtype=np.float32)
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs = probs / (probs.sum() + 1e-8)
        self.migration_likelihood = dict(zip(neighbors, probs))

    # -- 6. task migration ---------------------------------------------------
    def migrate_tasks(self, migration_action, all_servers):
        """
        migration_action: 6-vector, fraction of each queue's cpu/mem demand
        to migrate away this slot (independent per queue -- see
        networks.py). A destination is sampled per migrated task from
        `self.migration_likelihood`.
        """
        if not self.migration_likelihood:
            return  # no neighbours available/offering -- nothing to migrate

        cpu_frac, mem_frac = migration_action[:3], migration_action[3:]
        dest_ids = list(self.migration_likelihood.keys())
        dest_probs = np.array(list(self.migration_likelihood.values()))
        dest_probs = dest_probs / dest_probs.sum()

        # ASSUMPTION: a task is atomic (can't be "half migrated"), so the
        # per-queue cpu/mem fractions are averaged into a single fraction
        # of that queue's TASK COUNT to migrate this slot.
        combined_frac = (cpu_frac + mem_frac) / 2.0
        for q, frac in zip((LOW, MID, HIGH), combined_frac):
            queue = self.queues[q]
            n_migrate = int(round(float(frac) * len(queue)))
            for task in list(queue.tasks)[:n_migrate]:
                dest = int(np.random.choice(dest_ids, p=dest_probs))
                comm_time = self._migration_comm_time(task, dest)
                task.comm_time += comm_time
                task.ofloaded_server = dest
                task.origin_server = self.id
                # task.time_tolerance -= comm_time  # migration eats into the deadline
                queue.remove(task)
                all_servers[dest].queues[q].add(task)

                self.migrate_tasks_slot.append(task)
                self.stats["migrated"] += 1

    def _migration_comm_time(self, task, dest_id):
        speed = max(cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][self.id][dest_id], 1e-3)
        up_time = task.upload_size / cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][self.id][dest_id]
        down_time = task.download_size / cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][self.id][dest_id]
        return up_time + down_time  + cfg.LINK_LATENCY_S

    # -- 7. local execution ---------------------------------------------------

    def compute_reward_components(self):
        """
        Returns dict of the separate reward terms (see reward design from
        the chat discussion) so they can be logged individually as well as
        summed.

        Relies on `self._completed_this_slot` / `self._dropped_this_slot`
        containing ONLY tasks that finished/timed-out THIS slot. That
        invariant is now maintained by `advance_slot()` clearing both lists
        at the end of every slot (see module docstring for why this matters
        for training stability).
        """
        success_term = {
            "allocation_tasks": 0.0,
            "migration_tasks": 0.0,
            "sharing_tasks": [0.0] * len(cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]),  # placeholder for future expansion if sharing tasks are tracked,
        }
        violation_term = {
            "allocation_tasks": 0.0,
            "migration_tasks": 0.0,
            "sharing_tasks": [0.0] * len(cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]),
        }
        for t in self._completed_this_slot:
            norm_residual = max(t.residual_time, 0.0) / max(t.time_tolerance, 1e-6)
            norm_residual = min(norm_residual, 1.0)
            success_term["allocation_tasks"] += cfg.SUCCESS_WEIGHT[cfg.PRIORITIES[t.queue_type]] * norm_residual
            if t.ofloaded_server is not None:
                index = cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id].index(t.origin_server)
                success_term["sharing_tasks"][index] += cfg.SUCCESS_WEIGHT[cfg.PRIORITIES[t.queue_type]] * norm_residual
        for t in self._dropped_this_slot:
            violation_term["allocation_tasks"] -= cfg.VIOLATION_PENALTY[cfg.PRIORITIES[t.queue_type]]
            if t.ofloaded_server is not None:
                index = cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id].index(t.origin_server)
                violation_term["sharing_tasks"][index] -= cfg.VIOLATION_PENALTY[cfg.PRIORITIES[t.queue_type]]

        total_considered = len(self._completed_this_slot) + len(self._dropped_this_slot)
        completion_rate = (
            len(self._completed_this_slot) / total_considered if total_considered > 0 else 0.0
        )

        return {
            "success_term": success_term,
            "violation_term": violation_term,
            "completion_rate": completion_rate,
            "n_completed": len(self._completed_this_slot),
            "n_dropped": len(self._dropped_this_slot),
        }

    def execute(self, allocation_action):
        """
        allocation_action: 6-vector, proportions of CPU (first 3) and
        memory (last 3) allocated to the low/mid/high queues THIS SLOT.
        Sums to <= 1 within each resource (guaranteed ==1 by the softmax
        activation in networks.py). Returns the (unweighted) residual-time
        reward per queue; the caller applies PRIORITY_WEIGHT.
        """
        cpu_alloc, mem_alloc = allocation_action[:3], allocation_action[3:]
        residuals = {LOW: 0.0, MID: 0.0, HIGH: 0.0}
        
        for i, q in enumerate((LOW, MID, HIGH)):
            cpu_budget = cpu_alloc[i] * cfg.SERVERS[self.id]["cpu_hz"]
            mem_budget = mem_alloc[i] * cfg.SERVERS[self.id]["memory_mb"]
            queue = self.queues[q]
            n_tasks = max(len(queue), 1)
            # ASSUMPTION: this slot's cpu/mem budget for a queue is split
            # equally across the tasks currently waiting in it.
            # cpu_share = cpu_budget / n_tasks
            # mem_share = mem_budget / n_tasks

            finished = []
            total_exc_time = 0.0
            time_sum = 0.0
            time_cnt = 0
            for task in queue.tasks:
                if total_exc_time >= cfg.SLOT_DURATION_S:
                    # ASSUMPTION: if the first task can't finish this slot,
                    # none of the later tasks can either (FIFO).
                    break
                if task.remaining_cpu <= cpu_budget and task.remaining_memory <= mem_budget:
                    # task can finish this slot
                    exec_time = (task.remaining_cpu / max(cpu_budget, 1e-9)) * cfg.SLOT_DURATION_S if cpu_budget > 0 else float('inf')
                    
                    total_exc_time += exec_time
                    task.exec_time = total_exc_time
                    if task.residual_time >= 0:
                        time_sum += exec_time
                        time_cnt += 1
                        self.stats["success"] += 1
                        task.completed = True
                        norm_resi = max(task.residual_time, 0.0) / max(task.time_tolerance, 1e-6)  # only count positive residuals
                        norm_residual = min(norm_resi, 1.0)
                        residuals[q] += task.residual_time
                        self._completed_this_slot.append(task)
                    else:
                        self.stats["timeout"] += 1
                        task.dropped = True
                        residuals[q] += task.residual_time
                        self._dropped_this_slot.append(task)
                    finished.append(task)
                else:
                    task.time_tolerance -= cfg.SLOT_DURATION_S
                    if task.time_tolerance <= 0:
                        norm_resi = max(task.residual_time, 0.0) / max(task.time_tolerance, 1e-6)  # only count positive residuals
                        norm_residual = min(norm_resi, 1.0)
                        residuals[q] += task.residual_time
                        self.stats["timeout"] += 1
                        finished.append(task)
                        self._dropped_this_slot.append(task)
                    break  # ASSUMPTION: tasks are FIFO, so if the first task can't finish,

            for t in finished:
                queue.remove(t)
        
            self.avg_time[q] = time_sum / (time_cnt + 1e-12)

        return residuals

    def reward(self, residuals):
        return sum(cfg.PRIORITY_WEIGHT[q] * residuals[q] for q in (LOW, MID, HIGH))

    def advance_slot(self):
        """
        Advance to the next simulation slot.

        FIX: previously only `self.stats` was reset here, while
        `self._completed_this_slot`, `self._dropped_this_slot`, and
        `self.migrate_tasks_slot` accumulated for the ENTIRE rollout
        (they were only cleared in `reset()`). Since
        `compute_reward_components()` sums over `_completed_this_slot` /
        `_dropped_this_slot`, that bug made the reward at slot t include
        every task completed/dropped since slot 0 -- i.e. reward magnitude
        grew roughly linearly with `slot` across a 500-slot rollout. That
        non-stationarity is a primary cause of unstable training (drifting
        TD targets, exploding advantages). Clearing these per-slot lists
        here restores the intended "reward reflects only this slot"
        semantics.
        """
        self.slot += 1
        self.stats = dict(success=0, timeout=0, migrated=0, generated=0)
        self._completed_this_slot = []
        self._dropped_this_slot = []
        self.migrate_tasks_slot = []
        # self.avg_time = [0.0, 0.0, 0.0]
        # self.task_count = [0, 0, 0]