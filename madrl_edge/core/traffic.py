"""
Synthetic task arrival generator. Replace with real trace-driven arrivals.
CONFUSION: arrival rates / distributions are not specified anywhere in the
spec -- these are placeholder Poisson-ish arrivals for smoke-testing only.
"""
import random
from madrl_edge import config
from madrl_edge.core.tasks import Task
import pandas as pd
import os
import config as cfg

PRIORITY_ARRIVAL_RATE = {"HIGH": 2, "MID": 4, "LOW": 6}  # mean tasks/slot/queue
PRIORITY_TOLERANCE_RANGE = {"HIGH": (0.5, 2.0), "MID": (2.0, 6.0), "LOW": (6.0, 20.0)}
MAPPIG = {"HIGH": 2, "MID": 1, "LOW": 0}  # for indexing into load vector
print("Current working directory:", os.getcwd())
# CSV_DIR = "/mnt/5612DFC368CDC320/MSc Thesis/Code/MEC Implementation/Cooperated mec/LACT-Off/implementation/csv"
# task_number_dataframes = [pd.read_csv(f"{CSV_DIR}/tasks-{i}.csv", index_col=0) for i in range(config.NUM_SERVERS)]
task_number_dataframes = [pd.read_csv( cfg.TASK_DIR / f"tasks-{i}.csv", index_col=0) for i in range(cfg.NUM_SERVERS)]
for s in range(cfg.NUM_SERVERS):
    task_number_dataframes[s].index = range(len(task_number_dataframes[s]))
def generate_slot_arrivals(server_id: int, slot: int):
    tasks = []
    for priority in cfg.PRIORITIES:
        # n = max(0, int(random.gauss(PRIORITY_ARRIVAL_RATE[priority], 1.5)))
        n = int(task_number_dataframes[server_id].loc[slot, priority.lower()])
        # lo, hi = PRIORITY_TOLERANCE_RANGE[priority]
        for _ in range(n):
            lo, hi = cfg.TASK_CPU_RANGE[MAPPIG[priority]]
            cpu = random.uniform(lo, hi)
            lo, hi = cfg.TASK_MEM_RANGE[MAPPIG[priority]]
            mem = random.uniform(lo, hi)
            lo, hi = cfg.TASK_SIZE_RANGE[MAPPIG[priority]] 
            up = random.uniform(lo, hi)
            down = random.uniform(lo, hi)
            lo, hi = cfg.TASK_TOLERANCE_RANGE[MAPPIG[priority]]
            tol = random.uniform(lo, hi)
            tasks.append(Task(
                cpu_cycles=cpu,
                memory=mem,
                upload_bytes=up,
                download_bytes=down,
                tolerance=tol,
                priority=priority,
                arrival_slot=slot,
                origin_server=server_id,
            ))
    return tasks
