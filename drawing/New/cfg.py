"""Directory configuration.

If your project already has a `cfg` module with RESULT_DIR / GRAPH_DIR,
delete this file and keep yours -- the analysis scripts only need those two names.
"""
from pathlib import Path
import config as cfg
RESULT_DIR = cfg.RESULT_DIR   # <RESULT_DIR>/TIER_n/EXPERIMENT_NAME/*.csv
GRAPH_DIR = cfg.GRAPH_DIR     # all figures are saved here
