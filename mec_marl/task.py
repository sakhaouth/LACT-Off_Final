"""
task.py
=======
Basic data structures for tasks and per-priority task queues on an edge
server.

CONFUSED: the prompt says a queue's "load" is "the summation of cpu,
memory, time tolerance and the total number of tasks". This is read as:
for each queue, sum the CPU-cycle demand of all waiting tasks, sum their
memory demand, sum their time-tolerance, and count how many tasks are
waiting -> a 4-value load vector per queue. Concatenated over the 3
priority queues that gives the "vector of 3*4 size" mentioned later in the
prompt. If "summation" was meant differently (e.g. only counting tasks, or
averaging instead of summing), only `TaskQueue.compute_load()` below needs
to change.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional
import config as cfg

LOW, MID, HIGH = cfg.LOW, cfg.MID, cfg.HIGH
QUEUE_NAMES = {LOW: "low", MID: "mid", HIGH: "high"}

_id_counter = itertools.count()


@dataclass
class Task:
    """A single task waiting for CPU/memory allocation or migration."""

    id: int
    queue_type: int          # LOW, MID, or HIGH
    cpu_cycles: float        # required CPU cycles to complete the task
    memory: float            # required memory (arbitrary units, e.g. MB)
    upload_size: float       # bytes to upload (used for migration cost)
    download_size: float     # bytes to download (used for migration cost)
    time_tolerance: float    # deadline, in time-slot units
    created_at: int = 0       # slot index the task arrived in
    remaining_cpu: float = field(default=None)
    remaining_memory: float = field(default=None)
    exec_time: Optional[float] = None
    comm_time: float = 0.0
    completed: bool = False
    dropped: bool = False
    origin_server: Optional[int] = None  # server id where the task was generated
    ofloaded_server: Optional[int] = None  # server id where the task was migrated to


    def __post_init__(self):
        if self.remaining_cpu is None:
            self.remaining_cpu = self.cpu_cycles
        if self.remaining_memory is None:
            self.remaining_memory = self.memory

    @staticmethod
    def random_task(queue_type, slot, cfg, original_server=None):
        """Sample a random task for `queue_type` using ranges in `cfg`."""
        lo, hi = cfg.TASK_CPU_RANGE[queue_type]
        cpu = random.uniform(lo, hi)
        lo, hi = cfg.TASK_MEM_RANGE[queue_type]
        mem = random.uniform(lo, hi)
        lo, hi = cfg.TASK_SIZE_RANGE[queue_type]
        up = random.uniform(lo, hi)
        down = random.uniform(lo, hi)
        lo, hi = cfg.TASK_TOLERANCE_RANGE[queue_type]
        tol = random.uniform(lo, hi)
        return Task(
            id=next(_id_counter),
            queue_type=queue_type,
            cpu_cycles=cpu,
            memory=mem,
            upload_size=up,
            download_size=down,
            time_tolerance=tol,
            created_at=slot,
            origin_server=original_server,
        )
    @property
    def total_time(self) -> float:
        return (self.exec_time or 0.0) + self.comm_time

    @property
    def residual_time(self) -> float:
        """tolerance - actual time spent (comm + exec). Can be negative."""
        return self.time_tolerance - self.total_time


class TaskQueue:
    """FIFO-ish queue of tasks of a single priority level.

    A plain list is used instead of a deque so finished / migrated tasks
    (which are not necessarily at the front) can be removed directly.
    """

    def __init__(self, queue_type):
        self.queue_type = queue_type
        self.tasks = []

    def add(self, task: "Task"):
        self.tasks.append(task)

    def __len__(self):
        return len(self.tasks)

    def compute_load(self):
        """Return [total_cpu, total_memory, total_time_tolerance, count]."""
        if not self.tasks:
            return [0.0, 0.0, 0.0, 0.0]
        total_cpu = sum(t.remaining_cpu for t in self.tasks)
        total_mem = sum(t.remaining_memory for t in self.tasks)
        total_tol = sum(t.time_tolerance for t in self.tasks)
        cnt = float(len(self.tasks))
        norm_load = self.normalized_load(total_cpu, total_mem, total_tol, cnt)
        return norm_load
        # return [total_cpu, total_mem, total_tol, ]

    def remove(self, task: "Task"):
        self.tasks.remove(task)

    def pop_all(self):
        popped, self.tasks = self.tasks, []
        return popped
    def normalized_load(self, cpu, mem, tol, cnt):
        # cpu, mem, tol, cnt = self.compute_load()
        return [
            cpu / cfg.LOAD_NORM["cpu"],
            mem / cfg.LOAD_NORM["memory"],
            tol / cfg.LOAD_NORM["tolerance"],
            cnt / cfg.LOAD_NORM["count"],
        ]
