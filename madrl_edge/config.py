"""
Global configuration for the multi-agent edge-server task offloading system.

CONFUSION MARKERS: search for "CONFUSION:" throughout this codebase for every
place I had to make an assumption because the spec did not fully pin the value
down. Please review those before trusting the numbers.
"""

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

NUM_SERVERS = 6
# CONFUSION: adjacency is not specified. Using a simple ring + a couple of
# cross-links so every server has >=2 neighbours (avoids the old "BS3 empty
# NEIGHBORS" bug). Replace with your real topology.


#NEW-----------------------------------------------


DEV_CAPACITY = [
    [4.0e10, 128e9],  # BS0: City center MEC (≈40 GHz CPU, 128 GB RAM)
    [3.0e10,  96e9],  # BS1: Commercial district
    [2.0e10,  64e9],  # BS2: Urban baseline
    [1.5e10,  48e9],  # BS3: Residential area
    [1.0e10,  32e9],  # BS4: Suburban
    [5.0e9,   16e9],  # BS5: Sparse / rural
]
# NEIGHBORS = [
#             [1,2],
#             [0,2,3],
#             [0,1,4],
#             [1,4,5],
#             [2,3,5],
#             [3,4]
#         ]

NEIGHBORS = {
    
      0: [1,2],
      1: [0,2,3],
      2: [0,1,4],
      3: [1,4,5],
      4: [2,3,5],
      5: [3,4]
}


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



# NEIGHBORS = {
#     0: [1, 5],
#     1: [0, 2],
#     2: [1, 3],
#     3: [2, 4],
#     4: [3, 5],
#     5: [4, 0, 2],
# }

# ---------------------------------------------------------------------------
# Queues / priorities
# ---------------------------------------------------------------------------
PRIORITIES = ["LOW", "MID", "HIGH"]
NUM_QUEUES = len(PRIORITIES)
PRIORITY_INDEX = {"HIGH": 2, "MID": 1, "LOW": 0}

# Reward weighting per priority. CONFUSION: exact values are a design choice,
# not derivable from the spec ("multiply different factors ... for different
# priority queues" was stated but no numbers given). These are placeholders.
SUCCESS_WEIGHT = {"HIGH": 3.0, "MID": 1.5, "LOW": 1.0}
VIOLATION_PENALTY = {"HIGH": 6.0, "MID": 2.0, "LOW": 0.5}

# ---------------------------------------------------------------------------
# Load vector
# ---------------------------------------------------------------------------
# Per-queue load = [sum_cpu_cycles, sum_memory, sum_time_tolerance, num_tasks]
QUEUE_LOAD_DIM = 4
LOCAL_LOAD_DIM = NUM_QUEUES * QUEUE_LOAD_DIM  # 12, i.e. the "3*4" vector

# Normalisation caps for the four raw load components, used to keep the load
# vector roughly in [0, 1] before it hits the VRNN / actor.
# CONFUSION: these should be fit from real traffic statistics, not guessed.
LOAD_NORM = {
    "cpu": 1e10,       # total CPU cycles per queue, upper bound estimate
    "memory": 1e9,     # bytes
    "tolerance": 30.0, # seconds
    "count": 200.0,    # tasks
}

# ---------------------------------------------------------------------------
# VRNN (UPF global-state compressor)
# ---------------------------------------------------------------------------
# LOCAL_LOAD_DIM = 12
SEQ_LEN = 16
VRNN_SEQ_LEN = 16
VRNN_INPUT_DIM = LOCAL_LOAD_DIM * NUM_SERVERS       # each server contributes a 12-dim load
VRNN_HIDDEN_DIM = 64
VRNN_LATENT_DIM = 16
GLOBAL_LOAD_DIM = VRNN_LATENT_DIM       # what gets broadcast back to servers
# CONFUSION: "compressed global load" could mean (a) one shared vector for
# all servers, or (b) a per-server-conditioned vector. Implemented as (a):
# VRNN consumes the sequence/set of all server loads and produces one global
# summary vector broadcast identically to everyone. If you intended
# per-server personalization, the VRNN decoder needs to be conditioned on
# the querying server's own load too.

# ---------------------------------------------------------------------------
# Actor / action space
# ---------------------------------------------------------------------------
OBS_DIM = LOCAL_LOAD_DIM + GLOBAL_LOAD_DIM   # local load + global load given to actor
ACTION_DIM = 12
# action layout:
#   [0:6]  allocation: (cpu_high, mem_high, cpu_mid, mem_mid, cpu_low, mem_low)
#          cpu_high+cpu_mid+cpu_low == 1, mem_high+mem_mid+mem_low == 1
#          -> enforced via softmax in the Actor network, not learned "softly"
#   [6:12] migration: (cpu_frac_high, mem_frac_high, cpu_frac_mid, mem_frac_mid,
#          cpu_frac_low, mem_frac_low) each in [0,1], fraction of that queue's
#          CURRENT load to migrate away. Enforced via sigmoid.

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
# CONFUSION: per-server capacity is heterogeneous in real deployments but not
# specified. Using a flat default here; override per BaseStation instance.
DEFAULT_CPU_CAPACITY = 5e10       # cycles per slot
DEFAULT_MEMORY_CAPACITY = 4e9     # bytes

# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------
# CONFUSION: migration communication time model is not specified beyond
# "communication time will be added as additional time cost". Using a simple
# bandwidth model: comm_time = (upload_bytes + download_bytes) / link_bandwidth
# + fixed_link_latency. Replace with your real channel model if you have one
# (e.g. queueing delay, propagation distance, interference).
LINK_BANDWIDTH_BPS = 5e7   # 50 Mbps default inter-server link
LINK_LATENCY_S = 0.005     # 5 ms fixed latency per hop

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
SLOT_DURATION_S = 60.0  # CONFUSION: not specified; assumed 1s slots

# ---------------------------------------------------------------------------
# RL / training
# ---------------------------------------------------------------------------
GAMMA = 0.95
TAU = 0.01
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BUFFER_CAPACITY = 200_000
BATCH_SIZE = 256
WARMUP_STEPS = 5_000
ACTOR_UPDATE_DELAY = 2          # TD3-style delayed actor/target updates
CRITIC_WEIGHT_DECAY = 1e-5
GLOBAL_REWARD_MIX = 0.15        # lambda blending local vs mean-of-all reward


LOW, MID, HIGH = 2, 1, 0
TASK_CPU_RANGE = {"LOW": (5, 40), "MID": (5, 40), "HIGH": (5, 40)}
TASK_MEM_RANGE = {"LOW": (5, 40), "MID": (5, 40), "HIGH": (5, 40)}
TASK_SIZE_RANGE = {"LOW": (1e5, 5e5), "MID": (1e5, 5e5), "HIGH": (1e5, 5e5)}
TASK_TOLERANCE_RANGE = {"LOW": (6.0, 20.0), "MID": (2.0, 6.0), "HIGH": (0.5, 2.0)}
ROLLOUT_LEN = 1000
NUMBER_EPOCH = 20
