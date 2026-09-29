"""
config.py
=========
Mobile Edge Computing (MEC) Simulation Configuration
Region: Beijing Municipality, China
Bounding box: SW(39.4N, 115.7E) -> NE(41.1N, 117.4E)

This file is self-contained and directly executable. It defines:
  - 30 heterogeneous MEC servers with realistic Beijing locations
  - Four nested hierarchical deployment tiers: 6, 12, 18, 30 servers
  - A geographically-aware neighbor topology (NEIGHBORS_*) for each tier
  - A symmetric fiber bandwidth matrix (CONNECTION_SPEED_*) for each tier
  - Convenience lookup dictionaries (CPU, memory, coordinates, area types)

Running this file directly (`python config.py`) prints a summary report
and basic sanity checks (symmetry, connectivity, degree bounds).
"""

import math
from collections import deque

# ---------------------------------------------------------------------------
# 1. MASTER SERVER LIST (30 servers, IDs 0-29)
# ---------------------------------------------------------------------------
# The ordering of this list IS the hierarchy: servers 0-5 are the coarsest
# tier (6-server deployment), 0-11 the 12-server tier, 0-17 the 18-server
# tier, and 0-29 the full 30-server dense deployment. Because every smaller
# tier is a strict prefix of a larger one, each tier can be used completely
# independently without ever touching server locations.

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

assert len(SERVERS) == 30
assert [s["id"] for s in SERVERS] == list(range(30))

# ---------------------------------------------------------------------------
# 2. HIERARCHY TIER DEFINITIONS
# ---------------------------------------------------------------------------
TIERS = {6: list(range(6)), 12: list(range(12)), 18: list(range(18)), 30: list(range(30))}

# Target node degree (number of neighbors) by area type, used to build a
# realistic metropolitan fiber topology (2-5 neighbors per node; urban /
# CBD / technology-park nodes sit at the upper end, rural nodes at the
# lower end).
DEGREE_BY_TYPE = {
    "CBD": 5,
    "Technology Park": 5,
    "Airport": 4,
    "Transportation Hub": 4,
    "Industrial": 3,
    "Urban": 3,
    "Suburban": 3,
    "Rural": 2,
}

# Base backbone bandwidth (Mbps) by the pair of area types being connected.
# Symmetric lookup: order of the two types does not matter.
BASE_BANDWIDTH_MBPS = {
    frozenset(["CBD", "CBD"]): 10000,
    frozenset(["CBD", "Technology Park"]): 10000,
    frozenset(["Technology Park", "Technology Park"]): 10000,
    frozenset(["CBD", "Airport"]): 9000,
    frozenset(["Technology Park", "Airport"]): 9000,
    frozenset(["Airport", "Airport"]): 9000,
    frozenset(["CBD", "Transportation Hub"]): 8000,
    frozenset(["Technology Park", "Transportation Hub"]): 8000,
    frozenset(["Transportation Hub", "Transportation Hub"]): 8000,
    frozenset(["Airport", "Transportation Hub"]): 8000,
    frozenset(["CBD", "Urban"]): 6000,
    frozenset(["Technology Park", "Urban"]): 6000,
    frozenset(["Urban", "Urban"]): 5500,
    frozenset(["CBD", "Industrial"]): 5000,
    frozenset(["Technology Park", "Industrial"]): 5000,
    frozenset(["Industrial", "Industrial"]): 4500,
    frozenset(["Urban", "Industrial"]): 4500,
    frozenset(["Airport", "Industrial"]): 5500,
    frozenset(["Urban", "Suburban"]): 3000,
    frozenset(["CBD", "Suburban"]): 3200,
    frozenset(["Technology Park", "Suburban"]): 3200,
    frozenset(["Industrial", "Suburban"]): 2800,
    frozenset(["Suburban", "Suburban"]): 2500,
    frozenset(["Suburban", "Rural"]): 1200,
    frozenset(["Urban", "Rural"]): 1200,
    frozenset(["Rural", "Rural"]): 800,
    frozenset(["Industrial", "Rural"]): 1000,
    frozenset(["Airport", "Suburban"]): 3500,
    frozenset(["Airport", "Rural"]): 1200,
    frozenset(["Transportation Hub", "Urban"]): 6000,
    frozenset(["Transportation Hub", "Suburban"]): 3200,
    frozenset(["Transportation Hub", "Industrial"]): 5000,
    frozenset(["Transportation Hub", "Rural"]): 1000,
}


# ---------------------------------------------------------------------------
# 3. GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _server_by_id(sid):
    return SERVERS[sid]


# ---------------------------------------------------------------------------
# 4. NEIGHBOR TOPOLOGY CONSTRUCTION
# ---------------------------------------------------------------------------
MAX_DEGREE = 5
MIN_DEGREE = 2


def build_neighbors(server_ids):
    """
    Build a realistic, sparse, fully-connected neighbor graph over the given
    subset of server IDs, with every node's degree kept in [2, 5] (urban /
    CBD / technology-park nodes trend toward 5, rural nodes toward 2).

    Method:
      1. Compute a target degree per node from DEGREE_BY_TYPE, clamped to
         [MIN_DEGREE, MAX_DEGREE].
      2. Consider every possible edge sorted by geographic distance
         (ascending) and greedily add an edge whenever BOTH endpoints are
         still below their target degree. This mirrors how a real
         metropolitan fiber network is rolled out: closest points of
         presence are wired first, and no site is over-provisioned.
      3. Any node left below MIN_DEGREE (can happen for isolated outliers)
         is topped up by connecting it to its nearest remaining candidates
         that are still below MAX_DEGREE.
      4. Verify full connectivity via BFS. If more than one connected
         component remains (possible for far-flung rural nodes), bridge the
         components using their single closest inter-component pair of
         nodes, guaranteeing multi-hop reachability between any two nodes.
         Bridging is only ever needed for a handful of edges and, in rare
         cases, may push a node's degree one above the nominal cap in
         exchange for guaranteed connectivity -- connectivity is treated as
         a hard requirement that takes priority over the soft degree bound.
    """
    ids = list(server_ids)
    n = len(ids)

    dist = {(a, b): haversine_km(_server_by_id(a)["lat"], _server_by_id(a)["lon"],
                                  _server_by_id(b)["lat"], _server_by_id(b)["lon"])
             for a in ids for b in ids if a != b}

    target = {}
    for sid in ids:
        area_type = _server_by_id(sid)["area_type"]
        target[sid] = max(MIN_DEGREE, min(DEGREE_BY_TYPE.get(area_type, 3), MAX_DEGREE, n - 1))

    neighbors = {sid: set() for sid in ids}

    all_edges = sorted(
        [(dist[(a, b)], a, b) for a in ids for b in ids if a < b],
        key=lambda e: e[0],
    )

    # Pass 1: greedily wire nearest pairs while respecting target degree.
    for d, a, b in all_edges:
        if len(neighbors[a]) < target[a] and len(neighbors[b]) < target[b]:
            neighbors[a].add(b)
            neighbors[b].add(a)

    # Pass 2: top up any node still below MIN_DEGREE.
    for sid in ids:
        if len(neighbors[sid]) >= MIN_DEGREE:
            continue
        ranked = sorted([o for o in ids if o != sid], key=lambda o: dist[(sid, o)])
        for o in ranked:
            if len(neighbors[sid]) >= MIN_DEGREE:
                break
            if o in neighbors[sid]:
                continue
            if len(neighbors[o]) < MAX_DEGREE:
                neighbors[sid].add(o)
                neighbors[o].add(sid)

    # --- connectivity check & repair (bridge disconnected components) -----
    def connected_components():
        seen = set()
        comps = []
        for start in ids:
            if start in seen:
                continue
            comp = set()
            dq = deque([start])
            seen.add(start)
            while dq:
                cur = dq.popleft()
                comp.add(cur)
                for nb in neighbors[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        dq.append(nb)
            comps.append(comp)
        return comps

    comps = connected_components()
    while len(comps) > 1:
        c1, c2 = comps[0], comps[1]
        best_pair, best_d = None, float("inf")
        for a in c1:
            for b in c2:
                d = dist[(a, b)]
                if d < best_d:
                    best_d, best_pair = d, (a, b)
        a, b = best_pair
        neighbors[a].add(b)
        neighbors[b].add(a)
        comps = connected_components()

    return {sid: sorted(neighbors[sid]) for sid in ids}


def build_connection_speed(server_ids, neighbors):
    """Build a symmetric bandwidth matrix (Mbps) indexed 0..len(ids)-1 in
    the same order as `server_ids`. Bandwidth combines a base value for the
    pair of area types with a mild distance-based derating (longer fiber
    spans lose some effective throughput)."""
    ids = list(server_ids)
    idx = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    matrix = [[0] * n for _ in range(n)]

    for sid in ids:
        for nb in neighbors[sid]:
            if nb <= sid:
                continue  # fill each undirected edge once
            t1 = _server_by_id(sid)["area_type"]
            t2 = _server_by_id(nb)["area_type"]
            base = BASE_BANDWIDTH_MBPS.get(frozenset([t1, t2]), 1500)
            d = haversine_km(_server_by_id(sid)["lat"], _server_by_id(sid)["lon"],
                              _server_by_id(nb)["lat"], _server_by_id(nb)["lon"])
            # derate ~1.5% per km beyond 5 km, floor at 35% of base
            derate = max(0.35, 1 - 0.015 * max(0, d - 5))
            bw = int(round((base * derate) / 100.0)) * 100  # round to nearest 100 Mbps
            bw = max(bw, 500)
            i, j = idx[sid], idx[nb]
            matrix[i][j] = bw
            matrix[j][i] = bw
    return matrix


# ---------------------------------------------------------------------------
# 5. PRECOMPUTED TIER ARTIFACTS
# ---------------------------------------------------------------------------
SERVERS_6 = [s for s in SERVERS if s["id"] in TIERS[6]]
SERVERS_12 = [s for s in SERVERS if s["id"] in TIERS[12]]
SERVERS_18 = [s for s in SERVERS if s["id"] in TIERS[18]]
SERVERS_30 = SERVERS

NEIGHBORS_6 = build_neighbors(TIERS[6])
NEIGHBORS_12 = build_neighbors(TIERS[12])
NEIGHBORS_18 = build_neighbors(TIERS[18])
NEIGHBORS_30 = build_neighbors(TIERS[30])

CONNECTION_SPEED_6 = build_connection_speed(TIERS[6], NEIGHBORS_6)
CONNECTION_SPEED_12 = build_connection_speed(TIERS[12], NEIGHBORS_12)
CONNECTION_SPEED_18 = build_connection_speed(TIERS[18], NEIGHBORS_18)
CONNECTION_SPEED_30 = build_connection_speed(TIERS[30], NEIGHBORS_30)

# Convenience per-ID lookup dictionaries (span the full 0-29 ID space; a
# smaller tier simply uses the subset of keys it needs).
CPU_CAPACITY = {s["id"]: s["cpu_hz"] for s in SERVERS}          # cycles/sec
MEMORY_CAPACITY = {s["id"]: s["memory_mb"] for s in SERVERS}    # MB
COORDINATES = {s["id"]: (s["lat"], s["lon"]) for s in SERVERS}
AREA_TYPES = {s["id"]: s["area_type"] for s in SERVERS}
COVERAGE_RADIUS_KM = {s["id"]: s["coverage_km"] for s in SERVERS}

# Simulation region bounding box (as supplied)
REGION_BOUNDS = dict(sw=(39.4, 115.7), se=(39.4, 117.4), ne=(41.1, 117.4), nw=(41.1, 115.7))

# Bundled access point for simulation code
DEPLOYMENTS = {
    6:  dict(servers=SERVERS_6,  neighbors=NEIGHBORS_6,  connection_speed=CONNECTION_SPEED_6),
    12: dict(servers=SERVERS_12, neighbors=NEIGHBORS_12, connection_speed=CONNECTION_SPEED_12),
    18: dict(servers=SERVERS_18, neighbors=NEIGHBORS_18, connection_speed=CONNECTION_SPEED_18),
    30: dict(servers=SERVERS_30, neighbors=NEIGHBORS_30, connection_speed=CONNECTION_SPEED_30),
}


# ---------------------------------------------------------------------------
# 6. SELF-TEST / SUMMARY (runs only when executed directly)
# ---------------------------------------------------------------------------
def _bfs_is_connected(ids, neighbors):
    ids = list(ids)
    start = ids[0]
    seen = {start}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for nb in neighbors[cur]:
            if nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return len(seen) == len(ids)


def _matrix_is_symmetric(m):
    n = len(m)
    return all(m[i][j] == m[j][i] for i in range(n) for j in range(n))


if __name__ == "__main__":
    print(f"Beijing MEC Simulation Configuration ({len(SERVERS)} servers)")
    print("=" * 70)
    for tier in (6, 12, 18, 30):
        ids = TIERS[tier]
        neighbors = DEPLOYMENTS[tier]["neighbors"]
        matrix = DEPLOYMENTS[tier]["connection_speed"]
        degrees = [len(neighbors[i]) for i in ids]
        connected = _bfs_is_connected(ids, neighbors)
        symmetric = _matrix_is_symmetric(matrix)
        diag_zero = all(matrix[i][i] == 0 for i in range(len(ids)))
        print(f"\nTier {tier:>2} servers:")
        print(f"  degree range      : {min(degrees)}-{max(degrees)} "
              f"(avg {sum(degrees)/len(degrees):.2f})")
        print(f"  fully connected    : {connected}")
        print(f"  matrix symmetric   : {symmetric}")
        print(f"  matrix diag == 0   : {diag_zero}")
        edges = sum(degrees) // 2
        print(f"  edge count         : {edges}")
        print("neighbors")
        print(neighbors)
        print("matrix")
        print(matrix)
        print("deg")
        print(degrees)
        print("con")
        print(connected)
        print("sim")
        print(symmetric)
        print("diag")
        print(diag_zero)
    print("\nSample server record (ID 0):")
    for k, v in SERVERS[0].items():
        print(f"  {k:12}: {v}")