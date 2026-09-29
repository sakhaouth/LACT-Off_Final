from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent


RESULT_DIR = ROOT_DIR / "result_csv"
GRAPH_DIR = ROOT_DIR / "result_graphs"
MODEL_DIR = ROOT_DIR / "checkpoints"
TASK_PRODUCER_DIR = ROOT_DIR / "tasks_producer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIER_6 = 0
TIER_12 = 1
TIER_18 = 2
TIER_30 = 3
SERVER_COUNT = [6, 12, 18, 30]

CURRENT_TOPOLOGY = TIER_6
NUM_SERVERS = SERVER_COUNT[CURRENT_TOPOLOGY]
PREDICTOR_DATA_SLOT = (21*24*60)//15
PREDICTOR_TRAINING_END = (14*24*60)//15
PREDICTOR_TESTING_END = (21*24*60)//15
PDMA = 0
LACT_Off = 1
LACT_Off_MINUS = 2
PDMA_PLUS = 3
MINUS_PDMA = 4
MINUS_LACT_Off = 5
MODE_NAME = {
    PDMA : "PDMA",
    LACT_Off : "LACT-Off",
    LACT_Off_MINUS : "LACT-Off-",
    PDMA_PLUS : "PDMA+",
    MINUS_PDMA : "-PDMA",
    MINUS_LACT_Off : "-LACT_Off"
}
MODE_DES = {
    PDMA : "LSTM + VRNN + MADDPG",
    LACT_Off : "TimeLLM + Capacity based load aggregation + MTRL",
    LACT_Off_MINUS : "LSTM + Capacity based load aggregation + MTRL",
    PDMA_PLUS : "TimeLLM + VRNN + MADDPG",
    MINUS_PDMA : "No load predictor + VRNN + MADDPG",
    MINUS_LACT_Off : "No load predictor + Capacity based load aggregation + MTRL"
}


CPU_IMPORTANCE = 0.7
MEM_IMPORTANCE = 1 - CPU_IMPORTANCE
TOPO_NAME = {
    TIER_6 : "TIER_6",
    TIER_12 : "TIER_12",
    TIER_18 : "TIER_18",
    TIER_30 : "TIER_30",
}

CURRENT_RUNNIG_MODE = PDMA
TASK_DIR = ROOT_DIR / "tasks_producer" / TOPO_NAME[CURRENT_TOPOLOGY]
LOAD_DIR = ROOT_DIR / "load_files"
LINK_LATENCY_S = 0.005

CONFIG = {
    TIER_6: {
        "NEIGHBORS": {0: [1, 2, 3, 4], 1: [0, 2, 3, 4, 5], 2: [0, 1, 3, 5], 3: [0, 1, 2, 4], 4: [0, 1, 3], 5: [1, 2]},
        "CONNECTION_SPEED": [[0, 8600, 6700, 3700, 1300, 0], [8600, 0, 6300, 3200, 1600, 500], [6700, 6300, 0, 3200, 0, 500], [3700, 3200, 3200, 0, 1500, 0], [1300, 1600, 0, 1500, 0, 0], [0, 500, 500, 0, 0, 0]],

    },
    TIER_12: {
        "NEIGHBORS": {0: [1, 2, 6, 7, 9], 1: [0, 2, 6, 8, 10], 2: [0, 1, 8, 9], 3: [4, 7], 4: [3, 5], 5: [4, 11], 6: [0, 1, 7, 8, 10], 7: [0, 3, 6, 9], 8: [1, 2, 6], 9: [0, 2, 7], 10: [1, 6, 11], 11: [5, 10]},
        "CONNECTION_SPEED": [[0, 8600, 6700, 0, 0, 0, 9500, 4400, 0, 4900, 0, 0], [8600, 0, 6300, 0, 0, 0, 9500, 0, 2900, 0, 2100, 0], [6700, 6300, 0, 0, 0, 0, 0, 0, 5200, 1200, 0, 0], [0, 0, 0, 0, 1500, 0, 0, 3200, 0, 0, 0, 0], [0, 0, 0, 1500, 0, 500, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 500], [9500, 9500, 0, 0, 0, 0, 0, 4000, 2800, 0, 1700, 0], [4400, 0, 0, 3200, 0, 0, 4000, 0, 0, 3600, 0, 0], [0, 2900, 5200, 0, 0, 0, 2800, 0, 0, 0, 0, 0], [4900, 0, 1200, 0, 0, 0, 0, 3600, 0, 0, 0, 0], [0, 2100, 0, 0, 0, 0, 1700, 0, 0, 0, 0, 700], [0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 700, 0]],

    },
    TIER_18: {
        "NEIGHBORS": {0: [1, 6, 7, 12, 13], 1: [0, 2, 6, 12, 15], 2: [1, 8, 9, 12], 3: [4, 7], 4: [3, 5], 5: [4, 11], 6: [0, 1, 12, 13, 14], 7: [0, 3, 9], 8: [2, 10, 17], 9: [2, 7, 12], 10: [8, 11, 17], 11: [5, 10], 12: [0, 1, 2, 6, 9], 13: [0, 6, 14, 16], 14: [6, 13, 15, 16], 15: [1, 14, 16], 16: [13, 14, 15], 17: [8, 10]},
        "CONNECTION_SPEED": [[0, 8600, 0, 0, 0, 0, 9500, 4400, 0, 0, 0, 0, 9200, 7600, 0, 0, 0, 0], [8600, 0, 6300, 0, 0, 0, 9500, 0, 0, 0, 0, 0, 8700, 0, 0, 4500, 0, 0], [0, 6300, 0, 0, 0, 0, 0, 0, 5200, 1200, 0, 0, 8000, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1500, 0, 0, 3200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1500, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0], [9500, 9500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8700, 7900, 8000, 0, 0, 0], [4400, 0, 0, 3200, 0, 0, 0, 0, 0, 3600, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5200, 0, 0, 0, 0, 0, 0, 0, 1400, 0, 0, 0, 0, 0, 0, 800], [0, 0, 1200, 0, 0, 0, 0, 3600, 0, 0, 0, 0, 4800, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1400, 0, 0, 700, 0, 0, 0, 0, 0, 700], [0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 700, 0, 0, 0, 0, 0, 0, 0], [9200, 8700, 8000, 0, 0, 0, 8700, 0, 0, 4800, 0, 0, 0, 0, 0, 0, 0, 0], [7600, 0, 0, 0, 0, 0, 7900, 0, 0, 0, 0, 0, 0, 0, 7900, 0, 5700, 0], [0, 0, 0, 0, 0, 0, 8000, 0, 0, 0, 0, 0, 0, 7900, 0, 4700, 6000, 0], [0, 4500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4700, 0, 4300, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5700, 6000, 4300, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 700, 0, 0, 0, 0, 0, 0, 0]],
    },

    TIER_30: {
        "NEIGHBORS": {0: [6, 7, 12, 13, 24], 1: [6, 15, 22, 23, 24], 2: [8, 12, 21, 26], 3: [4, 9, 18], 4: [3, 18, 20], 5: [10, 27], 6: [0, 1, 13, 14, 23], 7: [0, 24, 25], 8: [2, 17, 26], 9: [3, 19, 24], 10: [5, 20, 29], 11: [28, 29], 12: [0, 2, 21, 22, 24], 13: [0, 6, 14, 16], 14: [6, 13, 16, 23], 15: [1, 16, 25], 16: [13, 14, 15], 17: [8, 26], 18: [3, 4, 25], 19: [9, 27], 20: [4, 10], 21: [2, 12, 22], 22: [1, 12, 21], 23: [1, 6, 14], 24: [0, 1, 7, 9, 12], 25: [7, 15, 18], 26: [2, 8, 17], 27: [5, 19], 28: [11, 29], 29: [10, 11, 28]},
        "CONNECTION_SPEED": [[0, 0, 0, 0, 0, 0, 9500, 4400, 0, 0, 0, 0, 9200, 7600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 9500, 0, 0, 0, 0, 0, 0, 0, 0, 4500, 0, 0, 0, 0, 0, 0, 5800, 6000, 8300, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 5200, 0, 0, 0, 8000, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 0, 0, 0, 0, 4700, 0, 0, 0], [0, 0, 0, 0, 1500, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 2400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1600, 0, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0], [9500, 9500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7900, 8000, 0, 0, 0, 0, 0, 0, 0, 0, 5900, 0, 0, 0, 0, 0, 0], [4400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4300, 3900, 0, 0, 0, 0], [0, 0, 5200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0, 0, 0, 4400, 0, 0, 0], [0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 5200, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 2300], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 900], [9200, 0, 8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 5800, 0, 9400, 0, 0, 0, 0, 0], [7600, 0, 0, 0, 0, 0, 7900, 0, 0, 0, 0, 0, 0, 0, 7900, 0, 5700, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8000, 0, 0, 0, 0, 0, 0, 7900, 0, 0, 6000, 0, 0, 0, 0, 0, 0, 5900, 0, 0, 0, 0, 0, 0], [0, 4500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4300, 0, 0, 0, 0, 0, 0, 0, 0, 3600, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5700, 6000, 4300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 0, 0], [0, 0, 0, 2400, 1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2800, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0], [0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2800, 0, 0, 0, 0, 0, 0, 0], [0, 5800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5800, 0, 0, 0, 0, 0, 0, 0, 0, 2800, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6000, 0, 0, 0, 0, 5900, 0, 0, 0, 0, 0, 0, 0, 5900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10000, 8300, 0, 0, 0, 0, 0, 4300, 0, 5200, 0, 0, 9400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3900, 0, 0, 0, 0, 0, 0, 0, 3600, 0, 0, 2800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 4700, 0, 0, 0, 0, 0, 4400, 0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2300, 900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0]],
    },
    
}

SERVERS = [
    # --- Tier 1: coarse city-wide coverage (IDs 0-5) -----------------------
    dict(id=0,  name="Beijing CBD",                    district="Chaoyang",
         lat=39.9086, lon=116.4568, area_type="CBD",
         cpu_hz=1.00e11, memory_mb=65536, coverage_km=8,
         desc="Central Business District; dense office towers, highest "
              "compute demand from finance, media and enterprise apps."),
    dict(id=1,  name="Zhongguancun Technology Park",   district="Haidian",
         lat=39.9836, lon=116.3164, area_type="Technology Park",
         cpu_hz=1.10e11, memory_mb=65536, coverage_km=8,
         desc="China's 'Silicon Valley'; universities and tech HQs, heavy "
              "AI/IoT edge-offload workloads."),
    dict(id=2,  name="Beijing Capital International Airport", district="Chaoyang/Shunyi",
         lat=40.0799, lon=116.5847, area_type="Airport",
         cpu_hz=9.00e10, memory_mb=32768, coverage_km=10,
         desc="Major international airport; passenger services, logistics "
              "tracking, aviation IoT."),
    dict(id=3,  name="Beijing Daxing International Airport", district="Daxing",
         lat=39.5098, lon=116.4109, area_type="Airport",
         cpu_hz=9.00e10, memory_mb=32768, coverage_km=10,
         desc="New mega-hub airport in the south of the region; growing "
              "passenger and cargo throughput."),
    dict(id=4,  name="Fangshan Suburban District",      district="Fangshan",
         lat=39.7387, lon=115.9931, area_type="Suburban",
         cpu_hz=3.00e10, memory_mb=8192, coverage_km=18,
         desc="South-western suburban/industrial fringe with light "
              "manufacturing and residential growth."),
    dict(id=5,  name="Miyun Mountainous Area",           district="Miyun",
         lat=40.6321, lon=116.8433, area_type="Rural",
         cpu_hz=1.50e10, memory_mb=4096, coverage_km=35,
         desc="Northern mountainous reservoir region; sparse population, "
              "low but non-zero demand (tourism, agriculture sensors)."),

    # --- Tier 2 additions: finer granularity (IDs 6-11) ---------------------
    dict(id=6,  name="Financial Street",                 district="Xicheng",
         lat=39.9139, lon=116.3610, area_type="CBD",
         cpu_hz=9.50e10, memory_mb=49152, coverage_km=6,
         desc="China's banking/regulatory headquarters cluster; low-latency "
              "financial transaction workloads."),
    dict(id=7,  name="Yizhuang Economic-Technological Development Area",
         district="Daxing/Yizhuang", lat=39.7952, lon=116.5074, area_type="Industrial",
         cpu_hz=6.50e10, memory_mb=16384, coverage_km=12,
         desc="State-level industrial/tech development zone; automotive, "
              "biotech and smart-manufacturing plants."),
    dict(id=8,  name="Shunyi Logistics & Industrial Park", district="Shunyi",
         lat=40.1301, lon=116.6547, area_type="Industrial",
         cpu_hz=6.00e10, memory_mb=16384, coverage_km=12,
         desc="Warehousing, freight and automotive-manufacturing corridor "
              "adjacent to the capital airport."),
    dict(id=9,  name="Tongzhou Sub-center",               district="Tongzhou",
         lat=39.9087, lon=116.6560, area_type="Urban",
         cpu_hz=5.50e10, memory_mb=16384, coverage_km=10,
         desc="Designated administrative sub-center of Beijing; new "
              "government offices and mixed-use development."),
    dict(id=10, name="Changping Suburban Tech Corridor",  district="Changping",
         lat=40.2206, lon=116.2314, area_type="Suburban",
         cpu_hz=3.50e10, memory_mb=8192, coverage_km=16,
         desc="University towns and light-tech industry belt north of the "
              "5th Ring Road."),
    dict(id=11, name="Yanqing Rural/Mountainous (Olympic Zone)", district="Yanqing",
         lat=40.4586, lon=115.9752, area_type="Rural",
         cpu_hz=1.80e10, memory_mb=4096, coverage_km=30,
         desc="Site of 2022 Winter Olympics venues; seasonal tourism load "
              "over an otherwise sparse mountainous area."),

    # --- Tier 3 additions: denser deployment (IDs 12-17) --------------------
    dict(id=12, name="Wangjing Technology & Business Area", district="Chaoyang",
         lat=40.0011, lon=116.4767, area_type="Technology Park",
         cpu_hz=8.50e10, memory_mb=32768, coverage_km=6,
         desc="Second major tech cluster (Korea-town/IT firms); dense "
              "residential-commercial mix."),
    dict(id=13, name="Beijing South Railway Station",      district="Fengtai",
         lat=39.8652, lon=116.3784, area_type="Transportation Hub",
         cpu_hz=7.00e10, memory_mb=16384, coverage_km=6,
         desc="High-speed rail terminus; massive transient passenger load, "
              "ticketing and surveillance analytics."),
    dict(id=14, name="Beijing West Railway Station",       district="Fengtai",
         lat=39.8949, lon=116.3219, area_type="Transportation Hub",
         cpu_hz=7.00e10, memory_mb=16384, coverage_km=6,
         desc="Largest conventional rail terminus in Asia at time of "
              "construction; continuous high passenger throughput."),
    dict(id=15, name="Shijingshan Industrial-Urban District", district="Shijingshan",
         lat=39.9142, lon=116.2229, area_type="Industrial",
         cpu_hz=4.50e10, memory_mb=16384, coverage_km=8,
         desc="Legacy steel-industry district under urban renewal into "
              "digital/media industries."),
    dict(id=16, name="Fengtai Urban District",              district="Fengtai",
         lat=39.8585, lon=116.2868, area_type="Urban",
         cpu_hz=4.50e10, memory_mb=8192, coverage_km=8,
         desc="Dense residential/commercial district south-west of the "
              "city core."),
    dict(id=17, name="Huairou Mountainous District",        district="Huairou",
         lat=40.3164, lon=116.6297, area_type="Rural",
         cpu_hz=1.80e10, memory_mb=4096, coverage_km=28,
         desc="Film-studio and tourism belt in the northern foothills."),

    # --- Tier 4 additions: dense metropolitan infrastructure (IDs 18-29) ----
    dict(id=18, name="Daxing District Center",              district="Daxing",
         lat=39.7280, lon=116.3357, area_type="Suburban",
         cpu_hz=3.20e10, memory_mb=8192, coverage_km=14,
         desc="Southern suburban administrative center undergoing rapid "
              "residential expansion."),
    dict(id=19, name="Pinggu Rural District",                district="Pinggu",
         lat=40.1443, lon=117.1119, area_type="Rural",
         cpu_hz=1.30e10, memory_mb=2048, coverage_km=32,
         desc="Eastern agricultural district (orchards); very low but "
              "non-negligible IoT/agri-sensor demand."),
    dict(id=20, name="Mentougou Mountainous District",       district="Mentougou",
         lat=39.9377, lon=115.7935, area_type="Rural",
         cpu_hz=1.30e10, memory_mb=2048, coverage_km=32,
         desc="Western mountainous district, former coal-mining area, now "
              "eco-tourism."),
    dict(id=21, name="Tiantongyuan Residential Cluster",     district="Changping/Chaoyang",
         lat=40.0819, lon=116.4204, area_type="Suburban",
         cpu_hz=3.80e10, memory_mb=8192, coverage_km=8,
         desc="One of Asia's largest residential super-blocks; heavy "
              "consumer mobile/video traffic at peak commute hours."),
    dict(id=22, name="Olympic Park Zone",                    district="Chaoyang",
         lat=39.9930, lon=116.3970, area_type="Urban",
         cpu_hz=5.00e10, memory_mb=16384, coverage_km=6,
         desc="2008 Olympic venues area; large-scale event and tourism "
              "traffic spikes."),
    dict(id=23, name="Haidian Secondary Business Zone",      district="Haidian",
         lat=39.9500, lon=116.3050, area_type="Urban",
         cpu_hz=5.50e10, memory_mb=16384, coverage_km=6,
         desc="Secondary commercial belt supporting the Zhongguancun tech "
              "core."),
    dict(id=24, name="Chaoyang Secondary CBD",                district="Chaoyang",
         lat=39.9200, lon=116.4900, area_type="CBD",
         cpu_hz=7.50e10, memory_mb=32768, coverage_km=6,
         desc="Extension of the CBD toward the eastern 4th Ring Road; "
              "embassies and multinational offices."),
    dict(id=25, name="Daxing Biomedical Industrial Base",     district="Daxing",
         lat=39.7700, lon=116.3400, area_type="Industrial",
         cpu_hz=5.50e10, memory_mb=16384, coverage_km=10,
         desc="Biomedical and pharmaceutical manufacturing park; "
              "compliance-sensitive, latency-tolerant batch workloads."),
    dict(id=26, name="Shunyi Airport Logistics Extension",    district="Shunyi",
         lat=40.1600, lon=116.7200, area_type="Industrial",
         cpu_hz=5.00e10, memory_mb=16384, coverage_km=10,
         desc="Cargo and cold-chain logistics extension of the airport "
              "corridor."),
    dict(id=27, name="Pinggu Eastern Border Rural",           district="Pinggu",
         lat=40.0700, lon=117.3500, area_type="Rural",
         cpu_hz=1.00e10, memory_mb=2048, coverage_km=35,
         desc="Far-eastern edge of the simulation boundary; minimal fixed "
              "population."),
    dict(id=28, name="Yanqing Extreme West Rural",            district="Yanqing",
         lat=40.5500, lon=115.7800, area_type="Rural",
         cpu_hz=1.00e10, memory_mb=2048, coverage_km=35,
         desc="North-western mountainous edge of the simulation boundary."),
    dict(id=29, name="Changping Northern Suburb/Tech",        district="Changping",
         lat=40.3000, lon=116.1500, area_type="Suburban",
         cpu_hz=3.20e10, memory_mb=8192, coverage_km=16,
         desc="Northern suburban belt with emerging light-industry parks."),
]


# ---------------------------------------------------------------------------
# Task queues / priorities
# ---------------------------------------------------------------------------
PRIORITIES = ["LOW", "MID", "HIGH"]

PRIO_MAP = {"LOW": 0,
          "MID": 1,
          "HIGH": 2,
          }
NUM_QUEUES = len(PRIORITIES)

LOW, MID, HIGH = 0, 1, 2
PRIORITY_INDEX = {"HIGH": 2, "MID": 1, "LOW": 0} 

SUCCESS_WEIGHT = {"HIGH": 3.0, "MID": 1.5, "LOW": 1.0}
VIOLATION_PENALTY = {"HIGH": 6.0, "MID": 2.0, "LOW": 0.5}

# Per-queue-type sampling ranges for new tasks (cpu cycles, memory, bytes,
# time-tolerance in slot units). ASSUMPTION: higher urgency -> tighter
# tolerance, similar resource needs across urgency levels.
TASK_CPU_RANGE = {LOW: (5, 40), MID: (5, 40), HIGH: (5, 40)}
TASK_MEM_RANGE = {LOW: (5, 40), MID: (5, 40), HIGH: (5, 40)}
TASK_SIZE_RANGE = {LOW: (1e5, 5e5), MID: (1e5, 5e5), HIGH: (1e5, 5e5)}
TASK_TOLERANCE_RANGE = {LOW: (6.0, 20.0), MID: (2.0, 6.0), HIGH: (0.5, 2.0)}


# Expected number of NEW tasks arriving per queue per slot (Poisson mean).
# Only present in draft A; kept since arrival-rate sampling needs it.
ARRIVAL_RATE = {LOW: 6.0, MID: 4.0, HIGH: 2.0}

VRNN_SEQ_LEN = 16
BUFFER_CAPACITY = 200_000
# ---------------------------------------------------------------------------
# Load vector (per-server local load fed into actors / VRNN)
# ---------------------------------------------------------------------------
# Per-queue load = [sum_cpu_cycles, sum_memory, sum_time_tolerance, num_tasks]
QUEUE_LOAD_DIM = 4
LOCAL_LOAD_DIM = NUM_QUEUES * QUEUE_LOAD_DIM  # 12, the "3*4" load vector
VRNN_HIDDEN_DIM = 64
VRNN_LATENT_DIM = 16
GLOBAL_LOAD_DIM = VRNN_LATENT_DIM
GLOBAL_REWARD_MIX = 0.15 
OBS_DIM = LOCAL_LOAD_DIM + GLOBAL_LOAD_DIM
# Normalisation caps for the four raw load components, used to keep the load
# vector roughly in [0, 1] before it hits the VRNN / actor.
# CONFUSION: these should be fit from real traffic statistics, not guessed.
LOAD_NORM = {
    # FIX: previously 1e10/1e9 -- many orders of magnitude larger than any
    # value raw_load() could ever actually produce given TASK_CPU_RANGE /
    # TASK_MEM_RANGE = (5, 40) per task. A full 200-task queue only sums to
    # ~8,000, so cpu/memory were always normalizing to ~0 regardless of real
    # congestion -- the actor and VRNN never saw real cpu/mem load, only
    # `count` (which happened to already be correctly scaled). Rescaled to
    # count * max-per-task so a genuinely full/loaded queue normalizes to
    # ~1.0. Still an estimate (see original CONFUSION note below) -- refit
    # from real traffic statistics if you have them.
    "cpu": 200.0 * 40,        # count * max TASK_CPU_RANGE per task
    "memory": 200.0 * 40,     # count * max TASK_MEM_RANGE per task
    "tolerance": 200.0 * 20,  # count * max TASK_TOLERANCE_RANGE (LOW queue, worst case)
    "count": 200.0,      # tasks
}
# FIX: caps how far normalized_load() can stray from the [-1, 1]-ish range
# the network is actually trained on, even under severe queue overload
# (hundreds of times over LOAD_NORM's assumed scale). See tasks.py.
LOAD_NORM_CLIP = 5.0


SLOT_DURATION_S = 900.0 #15 minuteC

SEQ_LEN = 16
ROLLOUT_LEN = 1000  # "T" in Algorithm 2 / rollout length for either trainer
NUM_ITERATIONS = 200
# =============================================================================
# VARIANT A -- Algorithm 2: Multi-Task A2C with Shared Critic
# (ALLOCATION / MIGRATION / SHARING actor-critics; from the first draft)
# =============================================================================
ALLOCATION, MIGRATION, SHARING = 0, 1, 2
NUM_DECISIONS = 3  # "M" in Algorithm 2
DECISION_NAMES = {ALLOCATION: "allocation", MIGRATION: "migration", SHARING: "sharing"}

# state = local load (NUM_QUEUES*4) concatenated with combined neighbor load
# (NUM_QUEUES*4) -- the "3*4 size" load vector from the prompt, doubled
# because both local and aggregated-neighbor loads are fed to the actors
# together.
A2C_STATE_DIM = NUM_QUEUES * 4 * 2   # 24
A2C_ACTION_DIM = NUM_QUEUES * 2      # 6 (3 queues x {cpu, memory})
MADDPG_ACTION_DIM = 12

NUM_WORKERS = 4         # "Nw" in Algorithm 2: parallel rollout replicas

A2C_GAMMA = 0.99
A2C_ACTOR_LR = 3e-4
A2C_CRITIC_LR = 1e-3
ALPHA = 0.5              # local-vs-shared advantage mix: alpha*A_local + (1-alpha)*A_shared
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
HIDDEN_DIM = 128


VRNN_SEQ_LEN = 16
VRNN_INPUT_DIM = LOCAL_LOAD_DIM * NUM_SERVERS       # each server contributes a 12-dim load
VRNN_HIDDEN_DIM = 64
VRNN_LATENT_DIM = 16

#=======================TIME-LLM+++++++++

# =====================================================
# Environment
# =====================================================
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# =====================================================
# Model
# =====================================================
LLM_MODEL = "TINYBERT"
LLM_LAYERS = 2
D_MODEL = 16
D_FF = 128
GRADIENT_CHECKPOINTING = True
USE_AMP = True
WARMUP_STEPS = 500
# =====================================================
# Training
# =====================================================
TRAIN_EPOCHS = 50
BATCH_SIZE = 64
LLM_MICRO_BATCH_SIZE = 8
LLM_LEARNING_RATE = 0.001
LSTM_LEARNING_RATE = 0.001
ITR = 1
MADDPG_BATCH_SIZE = 256

# =====================================================
# Dataset
# =====================================================
ROOT_PATH = "./CSV"
DATA_PATH = "./CSV/local_load_raw-4.csv"
DATA = "S_LOAD"
FEATURES = "M"
FREQ = "1min"
SEASONAL_PATTERNS = "every 1 minute"

# =====================================================
# Forecasting
# =====================================================
TASK_NAME = "long_term_forecast"
MODEL_ID = "ETTh1_512_96"
SEQ_LEN = 16
LABEL_LEN = 8
PRED_LEN = 1

# =====================================================
# Network
# =====================================================
ENC_IN = 7
DEC_IN = 7
C_OUT = 7
FACTOR = 3

# =====================================================
# Experiment
# =====================================================
IS_TRAINING = True
USE_LLM = True
USE_LSTM = True
DESCRIPTION = "Exp"
MODEL_COMMENT = "time-series-predictor"
N_HEADS = 8
PATCh_LEN = 16
STRIDE = 8
DROP_OUT = 0.1
LLM_DIM = 128
FACTOR = 3
PROMT_DOMAIN = 0
CONTENT = ""


#OTHERS
ACTOR_UPDATE_DELAY = 2  
TARGET_TAU = 0.01
TAU = 0.01
GAMMA = 0.99
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
ALPHA = 0.5  # local-vs-shared advantage mix: alpha*A_local + (1-alpha)*A_shared
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
CRITIC_WEIGHT_DECAY = 1e-5

# =====================================================
# MADDPG learning-stability additions
# =====================================================
# -- exploration noise schedule (previously hardcoded in train.py) ----------
NOISE_START = 0.3
NOISE_END = 0.02
NOISE_DECAY_STEPS = 100_000   # steps to anneal NOISE_START -> NOISE_END

# -- TD3-style target policy smoothing --------------------------------------
# Small noise injected into TARGET actor outputs when computing the critic's
# TD target. Reduces the systematic Q-overestimation that vanilla MADDPG
# (deterministic target policy) is prone to.
TARGET_SMOOTHING_STD = 0.1
TARGET_SMOOTHING_CLIP = 0.2

# -- reward normalization -----------------------------------------------
# Running (mean/std) normalization applied to rewards before they are used
# to form the critic's TD target. Raw success/violation terms scale with the
# (highly variable) number of tasks per queue per slot, so unnormalized
# rewards make the TD target's magnitude swing wildly between updates.
REWARD_NORM_CLIP = 10.0     # clip normalized reward to +/- this many std devs
REWARD_NORM_EPS = 1e-4

# -- backlog / queue stability -----------------------------------------
# FIX: previously nothing bounded how many tasks could pile up in a single
# priority queue, and nothing gave the agent reward signal about a growing
# backlog until a task's tolerance ran out and it got dropped. Together with
# a bug in _execute_queue (see environment.py) where only one task per
# queue per slot ever had its tolerance clock ticked, this let backlog (and,
# since the execution loop is O(len(queue.tasks)), per-episode wall-clock
# time) grow without bound over training -- exactly the failure mode in the
# "PDMA" training log (backlog 40 -> 1655, exc time 132s -> 490s over ~90
# episodes). MAX_QUEUE_LEN is a hard safety cap: even a badly-behaved policy
# can no longer make a single queue (and therefore a single training step)
# arbitrarily expensive. BACKLOG_PENALTY_WEIGHT gives the actor a small,
# dense, per-slot cost for letting queues sit long, well before tolerance
# actually expires.
MAX_QUEUE_LEN = 200          # matches the LOAD_NORM "full queue" assumption
BACKLOG_PENALTY_WEIGHT = 0.02