"""
BaseStation (single edge server) and the multi-agent environment orchestrating
one time slot across all servers.

Order of operations per slot (see CONFUSION note below on why this order was
chosen):
  1. every server computes its local load and sends it to neighbours + UPF
  2. UPF (frozen VRNN) compresses all loads -> broadcasts global load
  3. every server computes migration likelihood towards each neighbour
  4. actor(local_load, global_load) -> 12-dim action
  5. MIGRATION is executed first (using the migration fractions of the
     action, against CURRENT queue load)
  6. ALLOCATION is applied to the REMAINING (post-migration) queue load
  7. local execution proceeds using allocated CPU cycles
  8. reward computed per server (and migration outcome credit deferred to
     sender once receiver executes, next slot at the earliest)

CONFUSION: the spec doesn't state whether allocation happens against
pre-migration or post-migration load. I chose migrate-then-allocate because
allocating resources to tasks that are about to leave the server is wasteful;
if you intended simultaneous/independent computation, swap the order in
`BaseStation.step_slot` (just remove the `self._apply_migration_out(...)`
call before `self._apply_allocation(...)`).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import warnings
from collections import deque
from madrl_edge import config
from madrl_edge.core.tasks import Task, TaskQueue
from madrl_edge.core.load_utils import server_load_vector
from load_predictor.llm_inference import load_get_model, llm_inference
from load_predictor.lstm_inference import lstm_load_get_model, lstm_inference
from load_predictor.load_predictor_interface import Predictor

import config as cfg

@dataclass
class MigrationOrder:
    task: Task
    from_server: int
    to_server: int


class BaseStation:
    def __init__(self, server_id: int,
                 cpu_capacity: float = 0.0,
                 memory_capacity: float = 0.0, training_mode = False):
        self.id = server_id
        self.cpu_capacity = cfg.SERVERS[self.id]["cpu_hz"] if cpu_capacity <= 0.0 else cpu_capacity
        self.memory_capacity = cfg.SERVERS[self.id]["memory_mb"] if memory_capacity <= 0.0 else  memory_capacity
        self.neighbors = cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][self.id]
        self.queues: Dict[str, TaskQueue] = {p: TaskQueue(p) for p in config.PRIORITIES}

        # populated each slot
        self.last_local_load = None
        self.last_action = None
        self.migration_likelihood: Dict[int, float] = {}
        self._completed_this_slot: List[Task] = []
        self._dropped_this_slot: List[Task] = []
        if training_mode:
            self.predictor = None
        else:
            self.predictor = Predictor(self.id)
        self.avg_time = [0.0, 0.0 , 0.0]
        self.history = deque(
            [[0.0] * config.LOCAL_LOAD_DIM for _ in range(config.SEQ_LEN)],
            maxlen=config.SEQ_LEN
        )
    # ------------------------------------------------------------------

    def set_load_predictor(self, model, scaler,device, inference_func):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.inference_fun = inference_func
    
    def add_task(self, task: Task):
        self.queues[task.priority].add(task)


    def local_load_vector(self):
        local_raw_load = server_load_vector(self.queues)
        if self.predictor is None:
            return local_raw_load
        pred_load = self.predictor.predict(local_raw_load)

        self.last_local_load = None
        if cfg.CURRENT_RUNNIG_MODE == cfg.MINUS_PDMA or cfg.CURRENT_RUNNIG_MODE == cfg.MINUS_LACT_Off:
            self.last_local_load = local_raw_load
        else:
            self.last_local_load = list(pred_load)

        # FIX: the LSTM/TimeLLM predictor is a common NaN/Inf source (e.g. a
        # scaler dividing by zero variance on a queue that's been empty for
        # many slots). Catch it HERE, at the point of origin, rather than
        # letting it silently ride into obs_n / the replay buffer where it's
        # only detectable (and only fixable by skipping the whole batch)
        # much later in MADDPGCoordinator.update().
        arr = np.asarray(self.last_local_load, dtype=np.float64)
        if not np.isfinite(arr).all():
            bad_idx = np.where(~np.isfinite(arr))[0].tolist()
            warnings.warn(
                f"BaseStation {self.id}: non-finite predictor output at "
                f"local_load indices {bad_idx} (raw_load={local_raw_load}); "
                f"falling back to raw load for this slot."
            )
            self.last_local_load = list(local_raw_load)
        return self.last_local_load

    def migration_mask(self):
        """1.0 for each of the 6 migration action dims if this server HAS
        neighbours, else 0.0 -- prevents wasted exploration for isolated
        servers (the old BS3 zero-migration issue)."""
        has_neighbors = 1.0 if len(self.neighbors) > 0 else 0.0
        return np.full(config.NUM_QUEUES * 2, has_neighbors, dtype=np.float32)

    def compute_migration_likelihood(self, neighbor_loads: Dict[int, list]):
        """
        neighbor_loads: {neighbor_id: their 12-dim local load vector}
        Likelihood favors LESS loaded neighbours (inverse of total load),
        softmax-normalised over neighbours.
        CONFUSION: "migration likelihood" formula is not specified beyond
        "based on the shared load". Using inverse-load softmax as a simple,
        defensible default -- replace with whatever scoring you actually
        intend (e.g. could also factor in link distance/bandwidth).
        """
        if not neighbor_loads:
            self.migration_likelihood = {}
            return self.migration_likelihood

        scores = {}
        for nid, load in neighbor_loads.items():
            total_load = sum(load)  # lower = more available capacity
            scores[nid] = -total_load

        vals = np.array(list(scores.values()), dtype=np.float64)
        vals = vals - vals.max()
        exp = np.exp(vals)
        probs = exp / exp.sum()
        self.migration_likelihood = {
            nid: float(p) for nid, p in zip(scores.keys(), probs)
        }
        return self.migration_likelihood

    def pick_migration_target(self):
        if not self.migration_likelihood:
            return None
        nids = list(self.migration_likelihood.keys())
        probs = list(self.migration_likelihood.values())
        # CONFUSION: sampling vs argmax -- using argmax (most likely
        # neighbour) for determinism; sample if you want stochastic spreading.
        return nids[int(np.argmax(probs))]

    # ------------------------------------------------------------------
    def plan_migration(self, action: np.ndarray) -> List[MigrationOrder]:
        """
        Uses action[6:12] to decide, per queue, how many tasks to peel off
        and to which neighbour (single target per station per slot --
        CONFUSION: spec allows this to be per-queue but doesn't say whether
        different queues within the same server could pick different
        neighbours; here every queue on this server migrates to the SAME
        target, chosen once via `pick_migration_target`. Split-across-
        neighbours is a possible extension.)
        """
        target = self.pick_migration_target()
        orders: List[MigrationOrder] = []
        if target is None:
            return orders

        migration_fracs = action[6:12]
        for qi, priority in enumerate(config.PRIORITIES):
            cpu_frac = migration_fracs[qi * 2]
            mem_frac = migration_fracs[qi * 2 + 1]
            queue = self.queues[priority]
            raw_cpu, raw_mem, _, _ = queue.raw_load()
            cpu_target = cpu_frac * raw_cpu
            # memory target computed but only used as a soft/secondary signal
            # inside select_for_migration (see CONFUSION note in tasks.py)
            _mem_target = mem_frac * raw_mem
            selected = queue.select_for_migration(cpu_target, _mem_target)
            for t in selected:
                orders.append(MigrationOrder(task=t, from_server=self.id, to_server=target))
        return orders

    def remove_migrated(self, orders: List[MigrationOrder]):
        for o in orders:
            if o.task in self.queues[o.task.priority].tasks:
                self.queues[o.task.priority].remove(o.task)

    def receive_migrated(self, task: Task, comm_time: float):
        task.comm_time = comm_time
        self.queues[task.priority].add(task)

    # ------------------------------------------------------------------
    def get_avg_time(self):
        return self.avg_time
    def apply_allocation_and_execute(self, action: np.ndarray):
        """
        Applies action[0:6] to the CURRENT (post-migration) queue contents,
        computes execution time per task using the allocated CPU share, and
        resolves completion / drop for every task in every queue for this
        slot.

        CONFUSION: how allocated CPU cycles translate into per-TASK execution
        time within a queue isn't specified. Assumed: allocated CPU cycles
        for the queue are spent serially in priority-then-FIFO order across
        that queue's own tasks until either the slot budget or the CPU
        allocation runs out; tasks that don't get to run this slot remain
        queued for the next slot rather than being marked dropped, UNLESS
        their remaining tolerance is already <= 0, in which case they're
        dropped. Adjust if you intended a different resource-to-time model
        (e.g. proportional/fair sharing of instantaneous cycles across all
        tasks in the queue simultaneously).
        """
        self._completed_this_slot = []
        self._dropped_this_slot = []
        self.avg_time = [0.0, 0.0 , 0.0]
        alloc = action[0:6]
        for qi, priority in enumerate(cfg.PRIORITIES):
            cpu_share = alloc[qi * 2] * self.cpu_capacity
            mem_share = alloc[qi * 2 + 1] * self.memory_capacity
            queue = self.queues[priority]
            p = cfg.PRIORITY_INDEX[priority]
            self.avg_time[p] = self._execute_queue(queue, cpu_share, mem_share)
    def get_pred_local(self):
        return self.last_local_load
    def _execute_queue(self, queue: TaskQueue, cpu_budget: float, mem_budget: float):
        remaining_cpu = cpu_budget
        remaining_mem = mem_budget
        still_queued = []
        total_exc_time = 0.0
        total_time = 0
        total_cnt = 0
        for task in queue.tasks:
            if total_exc_time >= cfg.SLOT_DURATION_S:
                break  # slot time budget exhausted; remaining tasks wait for next slot
            if task.cpu_cycles <= remaining_cpu and task.memory <= remaining_mem:
                exec_time = task.cpu_cycles / max(cpu_budget, 1e-9) * cfg.SLOT_DURATION_S
                task.exec_time = exec_time
                total_exc_time += exec_time
                # remaining_cpu -= task.cpu_cycles
                # remaining_mem -= task.memory
                if task.residual_time >= 0:
                    task.completed = True
                    self._completed_this_slot.append(task)
                    total_time += exec_time
                    total_cnt += 1
                else:
                    task.dropped = True
                    self._dropped_this_slot.append(task)
            else:
                # not enough resource this slot
                task.tolerance -= cfg.SLOT_DURATION_S  # clock keeps ticking
                if task.tolerance <= 0:
                    task.dropped = True
                    self._dropped_this_slot.append(task)
                else:
                    still_queued.append(task)
                break  # stop processing this queue for this slot; remaining tasks wait for next slot
        queue.tasks = still_queued
        return total_time / (total_cnt + 1e-12)

    # ------------------------------------------------------------------
    def reward_components(self):
        """
        Returns dict of the separate reward terms (see reward design from
        the chat discussion) so they can be logged individually as well as
        summed.
        """
        success_term = 0.0
        violation_term = 0.0
        for t in self._completed_this_slot:
            norm_residual = max(t.residual_time, 0.0) / max(t.tolerance, 1e-6)
            norm_residual = min(norm_residual, 1.0)
            success_term += cfg.SUCCESS_WEIGHT[t.priority] * norm_residual
        for t in self._dropped_this_slot:
            violation_term += cfg.VIOLATION_PENALTY[t.priority]

        total_considered = len(self._completed_this_slot) + len(self._dropped_this_slot)
        completion_rate = (
            len(self._completed_this_slot) / total_considered if total_considered > 0 else 0.0
        )

        return {
            "success_term": success_term,
            "violation_term": -violation_term,
            "completion_rate": completion_rate,
            "n_completed": len(self._completed_this_slot),
            "n_dropped": len(self._dropped_this_slot),
        }
    def reset(self):
        self.predictor.reset()
        self.history = deque(
            [[0.0] * config.LOCAL_LOAD_DIM for _ in range(config.SEQ_LEN)],
            maxlen=config.SEQ_LEN
        )


class MultiAgentEdgeEnv:
    def __init__(self, num_servers=cfg.NUM_SERVERS, vrnn_encoder=None, training = False):
        self.num_servers = num_servers
        self.stations = []
        for i in range(num_servers):
            server = BaseStation(i, training_mode= training)
            # model, scaler, device = lstm_load_get_model(i)
            # server.set_load_predictor(model, scaler, device, lstm_inference)
            self.stations.append(server)

        self.vrnn_encoder = vrnn_encoder  # FrozenVRNNEncoder instance or None
        self.slot = 0

    def _global_load(self, all_local_loads):
        if self.vrnn_encoder is not None:
            z = self.vrnn_encoder.compress(all_local_loads)
            # FIX: an under-trained or mismatched vrnn.pt checkpoint (or a
            # history buffer still mostly zero-padded right after reset())
            # can emit NaN/Inf in z_final. This is a second, independent
            # potential NaN origin from the predictor above -- both feed
            # into the SAME obs_n, so both need their own guard.
            arr = np.asarray(z, dtype=np.float64)
            if not np.isfinite(arr).all():
                warnings.warn(
                    "FrozenVRNNEncoder.compress: non-finite global_load "
                    f"output {z}; falling back to zero vector for this slot."
                )
                z = [0.0] * cfg.GLOBAL_LOAD_DIM
            return z
        # CONFUSION: fallback when no VRNN is provided (e.g. for quick
        # smoke-testing without a pretrained checkpoint) -- just average.
        return list(np.mean(np.array(all_local_loads), axis=0))[: cfg.GLOBAL_LOAD_DIM] \
            if len(all_local_loads[0]) >= cfg.GLOBAL_LOAD_DIM \
            else list(np.mean(np.array(all_local_loads), axis=0)) + \
                 [0.0] * (cfg.GLOBAL_LOAD_DIM - len(all_local_loads[0]))

    def observe_all(self):
        temp_local_loads = [s.local_load_vector() for s in self.stations]
        local_loads = [s.get_pred_local() for s in self.stations]
        global_load = self._global_load(temp_local_loads)

        for s in self.stations:
            neighbor_loads = {n: local_loads[n] for n in s.neighbors}
            s.compute_migration_likelihood(neighbor_loads)

        obs_n = [np.array(local_loads[i] + list(global_load), dtype=np.float32)
                  for i in range(self.num_servers)]
        masks_n = [s.migration_mask() for s in self.stations]
        return obs_n, masks_n

    def step(self, action_n: List[np.ndarray]):
        # 1. plan + execute migrations (using CURRENT load, i.e. pre-allocation)
        all_orders = []
        for s, action in zip(self.stations, action_n):
            orders = s.plan_migration(action)
            all_orders.extend(orders)
            s.remove_migrated(orders)

        for o in all_orders:
            comm_time = self._comm_time(o.task, o.from_server, o.to_server)
            self.stations[o.to_server].receive_migrated(o.task, comm_time)

        # 2. allocation + local execution on remaining (post-migration) load
        for s, action in zip(self.stations, action_n):
            s.apply_allocation_and_execute(action)

        # 3. rewards
        reward_components_n = [s.reward_components() for s in self.stations]
        local_rewards = np.array([
            rc["success_term"] + rc["violation_term"] for rc in reward_components_n
        ])
        # CONFUSION: deferred migration credit (sender should get partial
        # credit/blame once a migrated task resolves at the receiver) is NOT
        # yet wired up here -- would require tracking task.origin_server
        # through to completion and crediting that server on a LATER slot.
        # This is flagged as a TODO rather than implemented, since it needs
        # a cross-slot bookkeeping structure not yet defined in this codebase.
        mean_reward = local_rewards.mean()
        blended_rewards = (
            (1 - cfg.GLOBAL_REWARD_MIX) * local_rewards
            + cfg.GLOBAL_REWARD_MIX * mean_reward
        )

        # 4. new arrivals would be injected here by the training loop /
        # traffic generator before the next observe_all() call.
        self.slot += 1
        next_obs_n, next_masks_n = self.observe_all()
        done_n = [False] * self.num_servers  # CONFUSION: episode-termination
        # condition isn't specified (fixed horizon? never-ending?) -- assumed
        # continuing task, handled by the training loop's episode length.

        info_n = reward_components_n
        cons_time = []
        for s in self.stations:
            cons_time.extend(s.avg_time)
        return next_obs_n, next_masks_n, list(blended_rewards), done_n, info_n, cons_time

    def _comm_time(self, task: Task, from_id: int, to_id: int) -> float:
        bytes_total = task.upload_bytes + task.download_bytes
        up_time = task.upload_bytes / cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][from_id][to_id]
        down_time = task.download_bytes / cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["CONNECTION_SPEED"][to_id][from_id]
        return up_time + down_time + cfg.LINK_LATENCY_S
        
    
    def reset(self):
        self.slot = 0
        self.vrnn_encoder.reset()
        for s in self.stations:
            s.queues = {p: TaskQueue(p) for p in cfg.PRIORITIES}
            s.last_local_load = None
            s.last_action = None
            s.migration_likelihood = {}
            s._completed_this_slot = []
            s._dropped_this_slot = []
            s.reset()
        return self.observe_all()