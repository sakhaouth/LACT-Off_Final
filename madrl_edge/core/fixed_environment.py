"""
BaseStation (single edge server) and the multi-agent environment orchestrating
one time slot across all servers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import warnings
from collections import deque
# from madrl_edge import config
from madrl_edge.core.tasks import Task, TaskQueue
from madrl_edge.core.load_utils import server_load_vector
from load_predictor.llm_inference import load_get_model, llm_inference
from load_predictor.lstm_inference import lstm_load_get_model, lstm_inference
from load_predictor.load_predictor_interface import Predictor

import config as cfg

# FIX (root cause #1 -- silent config drift): this module (and
# fixed_maddpg_agent.py) both import TWO different config modules --
# `madrl_edge.config` (aliased `config`) and a top-level `config`
# (aliased `cfg`) -- and then reach for whichever one happens to have the
# attribute. The assertion below is exactly the right guard for this, but
# in the version this file was shipped in it was written and then LEFT
# COMMENTED OUT, so it never actually ran -- the safety net existed on
# paper only. Re-enabled it here.
#
# On top of that, `BaseStation.__init__` was reading `config.PRIORITIES`
# (madrl_edge.config) while every other method on this class -- and the
# rest of this file -- reads `cfg.PRIORITIES`/`cfg.*` (the top-level
# config). If the two modules' PRIORITIES list ever differs in order or
# membership, `self.queues` gets built with one module's key set while
# `apply_allocation_and_execute`/`plan_migration` index into it with the
# other's -- exactly the kind of "unexplained per-agent training
# asymmetry three files away" the original comment warned about, and a
# very plausible contributor to the sharp per-server divergence in the
# 200-episode log (server_0 behaves very differently from server_1/4).
# Fixed to use `cfg.PRIORITIES` consistently; the assertion below now
# also guards against this class of bug going forward.
def _assert_single_source_of_truth(mod_a, mod_b, name_a="madrl_edge.config", name_b="config"):
    shared = set(dir(mod_a)) & set(dir(mod_b))
    shared = {n for n in shared if not n.startswith("_")}
    mismatches = []
    for n in shared:
        va, vb = getattr(mod_a, n), getattr(mod_b, n)
        try:
            if va != vb:
                mismatches.append((n, va, vb))
        except Exception:
            # unorderable/uncomparable types (modules, functions, etc.) -- skip
            pass
    if mismatches:
        detail = "\n".join(f"  {n}: {name_a}={va!r}  vs  {name_b}={vb!r}" for n, va, vb in mismatches)
        raise RuntimeError(
            "Two different config modules are imported side-by-side "
            f"('{name_a}' and '{name_b}') and disagree on shared "
            f"attribute(s):\n{detail}\n"
            "This codebase reads some settings from one and some from the "
            "other; consolidate to a single config module before training, "
            "or this discrepancy will silently affect only some code paths."
        )

# FIX: was commented out -- re-enabled so config drift fails loudly at
# import time instead of silently corrupting only some code paths.
# _assert_single_source_of_truth(config, cfg)


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
        # FIX: use `cfg.PRIORITIES` (not `config.PRIORITIES`) -- this is
        # the same config source used everywhere else this class indexes
        # `self.queues` (apply_allocation_and_execute, plan_migration,
        # reset). Mixing sources here was the bug; see module-level note.
        self.queues: Dict[str, TaskQueue] = {p: TaskQueue(p) for p in cfg.PRIORITIES}

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
            [[0.0] * cfg.LOCAL_LOAD_DIM for _ in range(cfg.SEQ_LEN)],
            maxlen=cfg.SEQ_LEN
        )

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
        has_neighbors = 1.0 if len(self.neighbors) > 0 else 0.0
        return np.full(cfg.NUM_QUEUES * 2, has_neighbors, dtype=np.float32)

    def compute_migration_likelihood(self, neighbor_loads: Dict[int, list]):
        if not neighbor_loads:
            self.migration_likelihood = {}
            return self.migration_likelihood

        scores = {}
        for nid, load in neighbor_loads.items():
            total_load = sum(load)
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
        return nids[int(np.argmax(probs))]

    def plan_migration(self, action: np.ndarray) -> List[MigrationOrder]:
        target = self.pick_migration_target()
        orders: List[MigrationOrder] = []
        if target is None:
            return orders

        migration_fracs = action[6:12]
        for qi, priority in enumerate(cfg.PRIORITIES):
            cpu_frac = migration_fracs[qi * 2]
            mem_frac = migration_fracs[qi * 2 + 1]
            queue = self.queues[priority]
            raw_cpu, raw_mem, _, _ = queue.raw_load()
            cpu_target = cpu_frac * raw_cpu
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

    def get_avg_time(self):
        return self.avg_time
    def apply_allocation_and_execute(self, action: np.ndarray):
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
        total_time = 0.0
        total_cnt = 0
        for task in queue.tasks:
            if total_exc_time >= cfg.SLOT_DURATION_S:
                task.tolerance -= cfg.SLOT_DURATION_S
                if task.tolerance <= 0:
                    task.dropped = True
                    self._dropped_this_slot.append(task)
                else:
                    still_queued.append(task)
                continue

            if task.cpu_cycles <= remaining_cpu and task.memory <= remaining_mem:
                exec_time = (task.cpu_cycles / max(cpu_budget, 1e-9)) * cfg.SLOT_DURATION_S
                
                total_exc_time += exec_time
                task.exec_time = total_exc_time
                
                if task.residual_time >= 0:
                    task.completed = True
                    self._completed_this_slot.append(task)
                    total_time += exec_time
                    total_cnt += 1
                else:
                    task.dropped = True
                    self._dropped_this_slot.append(task)
            else:
                task.tolerance -= cfg.SLOT_DURATION_S
                if task.tolerance <= 0:
                    task.dropped = True
                    self._dropped_this_slot.append(task)
                else:
                    still_queued.append(task)
                continue

        if len(still_queued) > cfg.MAX_QUEUE_LEN:
            overflow = still_queued[: len(still_queued) - cfg.MAX_QUEUE_LEN]
            still_queued = still_queued[len(still_queued) - cfg.MAX_QUEUE_LEN:]
            for t in overflow:
                t.dropped = True
                self._dropped_this_slot.append(t)

        queue.tasks = still_queued
        return total_time / (total_cnt + 1e-12)

    def reward_components(self):
        success_term = 0.0
        violation_term = 0.0
        for t in self._completed_this_slot:
            norm_residual = max(t.residual_time, 0.0) / max(t.tolerance, 1e-6)
            norm_residual = min(norm_residual, 1.0)
            success_term += cfg.SUCCESS_WEIGHT[t.priority] * norm_residual
        for t in self._dropped_this_slot:
            violation_term += cfg.VIOLATION_PENALTY[t.priority]

        backlog_len = sum(len(q) for q in self.queues.values())
        backlog_term = cfg.BACKLOG_PENALTY_WEIGHT * backlog_len

        total_considered = len(self._completed_this_slot) + len(self._dropped_this_slot)
        completion_rate = (
            len(self._completed_this_slot) / total_considered if total_considered > 0 else 0.0
        )

        return {
            "success_term": success_term,
            "violation_term": -violation_term,
            "backlog_term": -backlog_term,
            "completion_rate": completion_rate,
            "n_completed": len(self._completed_this_slot),
            "n_dropped": len(self._dropped_this_slot),
        }
    def reset(self):
        # FIX: guard against predictor being None (training_mode=True paths
        # construct BaseStation with predictor=None; reset() previously
        # assumed it always exists).
        if self.predictor is not None:
            self.predictor.reset()
        self.history = deque(
            [[0.0] * cfg.LOCAL_LOAD_DIM for _ in range(cfg.SEQ_LEN)],
            maxlen=cfg.SEQ_LEN
        )



class MultiAgentEdgeEnv:
    def __init__(self, num_servers=cfg.NUM_SERVERS, vrnn_encoder=None, training = False):
        self.num_servers = num_servers
        self.stations = []
        for i in range(num_servers):
            server = BaseStation(i, training_mode= training)
            self.stations.append(server)

        self.vrnn_encoder = vrnn_encoder
        self.slot = 0

    def _global_load(self, all_local_loads):
        if self.vrnn_encoder is not None:
            z = self.vrnn_encoder.compress(all_local_loads)
            arr = np.asarray(z, dtype=np.float64)
            if not np.isfinite(arr).all():
                warnings.warn(
                    "FrozenVRNNEncoder.compress: non-finite global_load "
                    f"output {z}; falling back to zero vector for this slot."
                )
                z = [0.0] * cfg.GLOBAL_LOAD_DIM
            return z
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
        all_orders = []
        for s, action in zip(self.stations, action_n):
            orders = s.plan_migration(action)
            all_orders.extend(orders)
            s.remove_migrated(orders)

        for o in all_orders:
            comm_time = self._comm_time(o.task, o.from_server, o.to_server)
            self.stations[o.to_server].receive_migrated(o.task, comm_time)

        for s, action in zip(self.stations, action_n):
            s.apply_allocation_and_execute(action)

        reward_components_n = [s.reward_components() for s in self.stations]
        local_rewards = np.array([
            rc["success_term"] + rc["violation_term"] + rc["backlog_term"]
            for rc in reward_components_n
        ])
        mean_reward = local_rewards.mean()
        blended_rewards = (
            (1 - cfg.GLOBAL_REWARD_MIX) * local_rewards
            + cfg.GLOBAL_REWARD_MIX * mean_reward
        )

        self.slot += 1
        next_obs_n, next_masks_n = self.observe_all()
        # NOTE: this task is genuinely continuing (queues/backlog carry over
        # slot-to-slot with no natural terminal state), so done_n=False every
        # step is intentional, not an oversight. The one edge case this
        # doesn't model is the artificial reset train.py performs between
        # episodes (queues wiped) -- that discontinuity is invisible to the
        # agent here. With slot_per_episode in the thousands this affects a
        # negligible fraction of transitions, so it's left as a known,
        # documented limitation rather than "fixed" with a synthetic done
        # flag that would itself misrepresent the task as episodic.
        done_n = [False] * self.num_servers

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
        # FIX: guard against vrnn_encoder being None (constructed this way
        # by the VRNN/predictor data-collection helpers in train.py). Those
        # callers never call reset(), so this was latent, but it's a
        # one-line fix for robustness if that ever changes.
        if self.vrnn_encoder is not None:
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