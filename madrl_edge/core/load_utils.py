"""
Server / queue load vector helpers.

Server load = concatenation of the 3 queues' 4-dim normalized load vectors,
in priority order [HIGH, MID, LOW] -> 12-dim vector, matching config.LOCAL_LOAD_DIM.
"""
from typing import List
from madrl_edge import config


def server_load_vector(queues) -> List[float]:
    """queues: dict priority -> TaskQueue, or list ordered [HIGH, MID, LOW]."""
    if isinstance(queues, dict):
        ordered = [queues[p] for p in config.PRIORITIES]
    else:
        ordered = queues
    vec = []
    for q in ordered:
        vec.extend(q.normalized_load())
    assert len(vec) == config.LOCAL_LOAD_DIM
    return vec
