"""
config.py
=========
All tunable constants for the environment and the Multi-Task A2C-with-
Shared-Critic trainer (Algorithm 2 in the prompt).

Two different "3"s show up in the scenario and they are NOT the same thing,
even though they happen to have the same value here:

  * NUM_QUEUES = 3    -> the three task-priority queues (low/mid/high) that
                          live *inside* every edge server.
  * NUM_DECISIONS = 3 -> the three decision types (ALLOCATION, MIGRATION,
                          SHARING) that Algorithm 2 calls "tasks" (M=3),
                          each with its own actor/critic pair.

CONFUSED: the prompt overloads the word "task" for both a unit of work
(low/mid/high-urgency Task objects) and an RL "task" in the multi-task-RL
sense (allocation/migration/sharing). I kept them as two separate constants
(NUM_QUEUES vs NUM_DECISIONS) to avoid silently conflating them. If your
intent was that the 3 RL "tasks" in Algorithm 2 actually correspond to the
3 priority queues instead of the 3 decision types, most of the per-decision
plumbing below (ALLOCATION/MIGRATION/SHARING) would need to move to a
per-queue split instead -- flag it and this can be restructured.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
NUM_SERVERS = 6

# CONFUSED: connection topology/speeds aren't specified beyond "connected by
# a network" -- assuming a fully-connected mesh with a random but FIXED
# bandwidth (arbitrary speed units) between every pair, generated once at
# import time (seeded) so every server agrees on the same numbers.
_rng = np.random.default_rng(0)
# CONNECTION_SPEED = _rng.uniform(50.0, 200.0, size=(NUM_SERVERS, NUM_SERVERS))
# np.fill_diagonal(CONNECTION_SPEED, 0.0)
# CONNECTION_SPEED = (CONNECTION_SPEED + CONNECTION_SPEED.T) / 2  # symmetric

# ASSUMPTION: fully connected -- restrict this dict's lists for a sparser
# topology (e.g. only physically nearby servers).
# NEIGHBORS = {s: [n for n in range(NUM_SERVERS) if n != s] for s in range(NUM_SERVERS)}
# NEIGHBORS = {
#     0: [1, 5],
#     1: [0, 2],
#     2: [1, 3],
#     3: [2, 4],
#     4: [3, 5],
#     5: [4, 0, 2],
# }
# ---------------------------------------------------------------------------
# Server resources
# ---------------------------------------------------------------------------
# CPU_CAPACITY = 5e10      # total CPU cycles/slot a server can allocate
# MEMORY_CAPACITY = 4e9  # total memory a server can allocate



#NEW----------------------------
DEV_CAPACITY = [
    [4.0e10, 128e9],  # BS0: City center MEC (≈40 GHz CPU, 128 GB RAM)
    [3.0e10,  96e9],  # BS1: Commercial district
    [2.0e10,  64e9],  # BS2: Urban baseline
    [1.5e10,  48e9],  # BS3: Residential area
    [1.0e10,  32e9],  # BS4: Suburban
    [5.0e9,   16e9],  # BS5: Sparse / rural
]
NEIGHBORS = {
    
      0: [1,2],
      1: [0,2,3],
      2: [0,1,4],
      3: [1,4,5],
      4: [2,3,5],
      5: [3,4]
}
PRIORITIES = ["LOW", "MID", "HIGH"]
SUCCESS_WEIGHT = {"HIGH": 3.0, "MID": 1.5, "LOW": 1.0}
VIOLATION_PENALTY = {"HIGH": 6.0, "MID": 2.0, "LOW": 0.5}
    # 5 ms fixed latency per hop
LINK_LATENCY_S = 0.005

CONNECTION_SPEED = [
    [0,    5000, 4000,    0,    0,    0],
    [5000,    0, 3500, 2500,    0,    0],
    [4000, 3500,    0,    0, 2000,    0],
    [0,    2500,    0,    0, 2000, 1200],
    [0,       0, 2000, 2000,    0, 1000],
    [0,       0,    0, 1200, 1000,    0]
]
# CONNECTION_SPEED = [
#     # BS0 (City Center) → BS1, BS2
#     [
#         [2000, 5000],   # to BS1 (fiber, very high DL)
#         [1500, 4000],   # to BS2
#     ],

#     # BS1 (Commercial) → BS0, BS2, BS3
#     [
#         [2000, 5000],   # to BS0
#         [1200, 3500],   # to BS2
#         [800,  2500],   # to BS3
#     ],

#     # BS2 (Urban baseline) → BS0, BS1, BS4
#     [
#         [1500, 4000],   # to BS0
#         [1200, 3500],   # to BS1
#         [600,  2000],   # to BS4
#     ],

#     # BS3 (Residential) → BS1, BS4, BS5
#     [
#         [800,  2500],   # to BS1
#         [700,  2000],   # to BS4
#         [400,  1200],   # to BS5
#     ],

#     # BS4 (Suburban) → BS2, BS3, BS5
#     [
#         [600,  2000],   # to BS2
#         [700,  2000],   # to BS3
#         [300,  1000],   # to BS5
#     ],

#     # BS5 (Sparse / rural) → BS3, BS4
#     [
#         [400,  1200],   # to BS3
#         [300,  1000],   # to BS4
#     ]
# ]

# ---------------------------------------------------------------------------
# Task queues / generation
# ---------------------------------------------------------------------------
LOW, MID, HIGH = 2, 1, 0
NUM_QUEUES = 3

# CONFUSED: priority-weighting multipliers for the residual-time reward
# aren't given numerically in the prompt ("multiply different factors...
# for different types priority queues") -- using HIGH > MID > LOW as the
# only sensible default; tune freely.
PRIORITY_WEIGHT = {LOW: 1.0, MID: 2.0, HIGH: 4.0}

# Per-queue-type sampling ranges for new tasks (cpu cycles, memory, bytes,
# time-tolerance in slot units). ASSUMPTION: higher urgency -> tighter
# tolerance, similar resource needs across urgency levels.
TASK_CPU_RANGE = {LOW: (5, 40), MID: (5, 40), HIGH: (5, 40)}
TASK_MEM_RANGE = {LOW: (5, 40), MID: (5, 40), HIGH: (5, 40)}
TASK_SIZE_RANGE = {LOW: (1e5, 5e5), MID: (1e5, 5e5), HIGH: (1e5, 5e5)}
TASK_TOLERANCE_RANGE = {LOW: (6.0, 20.0), MID: (2.0, 6.0), HIGH: (0.5, 2.0)}

# Expected number of NEW tasks arriving per queue per slot (Poisson mean).
ARRIVAL_RATE = {LOW: 6.0, MID: 4.0, HIGH: 2.0}

# ---------------------------------------------------------------------------
# RL / Algorithm 2 hyperparameters
# ---------------------------------------------------------------------------
ALLOCATION, MIGRATION, SHARING = 0, 1, 2
NUM_DECISIONS = 3  # "M" in Algorithm 2
DECISION_NAMES = {ALLOCATION: "allocation", MIGRATION: "migration", SHARING: "sharing"}

LOCAL_STATE_DIM = 12
# state = local load (NUM_QUEUES*4) concatenated with combined neighbor load
# (NUM_QUEUES*4) -- matches the "3*4 size" load vector from the prompt,
# doubled because both local and aggregated-neighbor loads are fed to the
# actors together ("calculates its load ... share this with its neighbour
# ... combine the neighbor's load ... to this 3 actor networks").
STATE_DIM = NUM_QUEUES * 4 * 2
ACTION_DIM = NUM_QUEUES * 2  # 3 queues x {cpu, memory}

NUM_WORKERS = 4       # "Nw" in Algorithm 2: parallel rollout replicas
ROLLOUT_LEN = 500  # "T" in Algorithm 2
NUM_ITERATIONS = 200   # outer loop count ("k" in Algorithm 2)

GAMMA = 0.99
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
ALPHA = 0.5  # local-vs-shared advantage mix: alpha*A_local + (1-alpha)*A_shared
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0

HIDDEN_DIM = 128

SLOT_DURATION_S = 60.0

SEQ_LEN = 16
LOAD_NORM = {
    "cpu": 1e10,       # total CPU cycles per queue, upper bound estimate
    "memory": 1e9,     # bytes
    "tolerance": 30.0, # seconds
    "count": 200.0,    # tasks
}