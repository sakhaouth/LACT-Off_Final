"""
Task and TaskQueue primitives.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import itertools

from madrl_edge import config
import config as cfg
_id_counter = itertools.count()


@dataclass
class Task:
    cpu_cycles: float          # required CPU cycles to complete
    memory: float               # required memory (bytes)
    upload_bytes: float         # size of data to upload if migrated
    download_bytes: float       # size of result to download if migrated
    tolerance: float            # time tolerance (seconds) before deadline
    priority: str                # "HIGH" | "MID" | "LOW"
    arrival_slot: int = 0
    origin_server: int = -1      # CONFUSION: needed for deferred migration
                                  # credit assignment (see reward.py). Must be
                                  # set to the server id where the task was
                                  # originally generated, not the current host.
    task_id: int = field(default_factory=lambda: next(_id_counter))

    # filled in during execution
    exec_time: Optional[float] = None
    comm_time: float = 0.0
    completed: bool = False
    dropped: bool = False

    @property
    def total_time(self) -> float:
        return (self.exec_time or 0.0) + self.comm_time

    @property
    def residual_time(self) -> float:
        """tolerance - actual time spent (comm + exec). Can be negative."""
        return self.tolerance - self.total_time


class TaskQueue:
    """A single priority queue on one BaseStation."""

    def __init__(self, priority: str):
        self.priority = priority
        self.tasks: List[Task] = []

    def add(self, task: Task):
        self.tasks.append(task)

    def remove(self, task: Task):
        self.tasks.remove(task)

    def clear(self):
        self.tasks = []

    def __len__(self):
        return len(self.tasks)

    # ------------------------------------------------------------------
    def raw_load(self):
        """Un-normalised [sum_cpu, sum_memory, sum_tolerance, count]."""
        if not self.tasks:
            return [0.0, 0.0, 0.0, 0.0]
        cpu = sum(t.cpu_cycles for t in self.tasks)
        mem = sum(t.memory for t in self.tasks)
        tol = sum(t.tolerance for t in self.tasks)
        cnt = float(len(self.tasks))
        return [cpu, mem, tol, cnt]

    def normalized_load(self):
        cpu, mem, tol, cnt = self.raw_load()
        return [
            cpu / cfg.LOAD_NORM["cpu"],
            mem / cfg.LOAD_NORM["memory"],
            tol / cfg.LOAD_NORM["tolerance"],
            cnt / cfg.LOAD_NORM["count"],
        ]

    # ------------------------------------------------------------------
    def select_for_migration(self, cpu_target: float, mem_target: float):
        """
        Greedily pick tasks to migrate until the CPU target is (approximately)
        met, using memory as a secondary/soft target.

        CONFUSION: the action gives two INDEPENDENT targets (cpu fraction,
        memory fraction) but tasks bundle cpu+memory together, so both
        targets generally cannot be hit exactly by any subset of tasks. This
        is an inherent under-determinism in the 6-value migration action
        design (see chat discussion). Policy used here: sort tasks by
        cpu_cycles descending (largest first) and keep adding until the CPU
        target is reached or exceeded; memory is whatever comes along with
        it. If you instead want memory-priority selection, swap the sort key
        or blend a weighted score.
        """
        if cpu_target <= 0 or not self.tasks:
            return []
        candidates = sorted(self.tasks, key=lambda t: t.cpu_cycles, reverse=True)
        selected = []
        cum_cpu = 0.0
        for t in candidates:
            if cum_cpu >= cpu_target:
                break
            selected.append(t)
            cum_cpu += t.cpu_cycles
        return selected
