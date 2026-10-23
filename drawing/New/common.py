"""Shared helpers for all analysis scripts (file lookup, loaders, aggregation, plotting).

Everything that depends on file naming lives here, so if your file names differ
slightly from the assumptions below, this is the only file you need to edit
(see `exp_dir`, `locate` and the `core` patterns in `load_performance` / `load_training`).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import drawing.New.cfg as cfg  # noqa: E402

CATEGORIES = ("low", "mid", "high")  # task priority categories
DEFAULT_SENS = (0, 0, 0)


# --------------------------------------------------------------------------- #
# Experiment / naming helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Exp:
    """An experiment such as 'PDMA', '-LACT_Off', 'LACT_Off-' or '[HOMO]PDMA'."""
    name: str
    homo: bool = False

    @property
    def label(self) -> str:
        return f"[HOMO]{self.name}" if self.homo else self.name

    @property
    def family(self) -> str:
        return "pdma" if "PDMA" in self.name.upper() else "lact"

    @classmethod
    def parse(cls, text: str) -> "Exp":
        t = text.strip()
        if t.upper().startswith("[HOMO]"):
            return cls(t[6:], True)
        return cls(t)


def tier_name(t) -> str:
    s = str(t)
    return f"TIER_{int(s)}" if s.isdigit() else s


def n_servers(tier) -> int:
    return int(tier_name(tier).split("_")[-1])


def sens_tag(sens) -> str:
    c, m, s = sens
    return f"SENSITIVITY{c}_{m}_{s}"

def tier_to_idx(n):
    if n == 6:
        return 0
    if n == 12:
            return 1
    if n == 18:
            return 2
    if n == 30:
            return 3

def exp_dir(tier, exp_name: str) -> Path:
    """<RESULT_DIR>/<TIER_n>/<EXPERIMENT_NAME>  (HOMO runs are assumed to live in the same folder)."""
    return Path(cfg.RESULT_DIR) / tier_name(tier) / exp_name


def locate(directory, core, homo, required=True):
    def natural(p):
        return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]

    # escape [ ] ? literally, but keep '*' as a wildcard
    pattern = "*" + glob.escape(core).replace("[*]", "*")
    hits = [p for p in Path(directory).glob(pattern) if ("HOMO" in p.name.upper()) == homo]
    hits.sort(key=natural)
    if not hits and required:
        raise FileNotFoundError(f"No file matching '{pattern}' (homo={homo}) in {directory}")
    return hits


def read_csv(path: Path) -> pd.DataFrame:
    """All logs have the iteration number in the first column (named 'episode' or unnamed)."""
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(str).str.strip()
    df.index.name = "iteration"
    return df


# --------------------------------------------------------------------------- #
# Alignment / math helpers
# --------------------------------------------------------------------------- #
def align_mean(frames: list[pd.DataFrame], mode: str = "common") -> pd.DataFrame:
    """Element-wise mean of several DataFrames whose iteration indices may differ.

    mode='common': keep only iterations present in every frame (no tail artefacts).
    mode='union' : keep all iterations; each is averaged over the frames that have it.
    """
    if not frames:
        raise ValueError("nothing to align")
    if mode == "common":
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.intersection(f.index)
        frames = [f.loc[idx] for f in frames]
    elif mode == "union":
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.union(f.index)
        frames = [f.reindex(idx) for f in frames]
    return pd.concat(frames).groupby(level=0).mean().sort_index()


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return s * 0.0
    return (s - lo) / (hi - lo)


def smooth(s: pd.Series, window: int = 1) -> pd.Series:
    return s if window <= 1 else s.rolling(window, min_periods=1).mean()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def perf_frame(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-iteration performance metrics from one performance log (one PDMA run / one LACT_Off worker)."""
    ids = [b for b in range(n) if f"success_{b}" in df.columns and f"generation_{b}" in df.columns]
    if not ids:
        raise ValueError("no success_*/generation_* columns found")
    succ = df[[f"success_{b}" for b in ids]].astype(float)
    gen = df[[f"generation_{b}" for b in ids]].astype(float)

    out = pd.DataFrame(index=df.index)
    out["total_completed"] = succ.sum(axis=1)
    out["generated"] = gen.sum(axis=1)
    # completion_rate = sum(success_b) / sum(generation_b)   [%]
    out["completion_rate"] = 100 * out["total_completed"] / out["generated"].where(out["generated"] > 0)
    # mean over servers of the per-server completion rate (used by SENSITIVITY)
    per_server = succ.values / gen.where(gen > 0).values
    out["completion_rate_server_mean"] = 100 * pd.DataFrame(per_server, index=df.index).mean(axis=1)
    # average over servers of the per-server task time of each priority category
    for cat in CATEGORIES:
        cols = [f"{cat}_time_{b}" for b in ids if f"{cat}_time_{b}" in df.columns]
        out[f"{cat}_time"] = df[cols].astype(float).mean(axis=1) if cols else np.nan
    return out


def load_performance(exp: Exp, tier, sens=DEFAULT_SENS, align: str = "common") -> pd.DataFrame:
    """Per-iteration performance table for one experiment.

    PDMA     : one file  ..._TOPO{n}_training_log.csv
    LACT_Off : many files ..._TOPO{n}_performance_parameter_worker{w}.csv, averaged over workers
           
                   (mean of per-worker totals / ratios, exactly as in the description).
    """
    # print(f"all {exp} {tier} {sens} {align}")
    tier, n = tier_name(tier), n_servers(tier)
    d, tag = exp_dir(tier, exp.name), sens_tag(sens)
    if exp.family == "pdma":
        paths = locate(d, f"[{tag}]_{exp.name}_TOPO[{tier_to_idx(n)}]_training_log.csv", exp.homo)[:1]
    else:
        paths = locate(d, f"[{tag}]_{exp.name}_TOPO[{tier_to_idx(n)}]_performance_parameter_worker*.csv", exp.homo)
    # print(paths)
    return align_mean([perf_frame(read_csv(p), n) for p in paths], align)


def load_training(exp: Exp, tier, sens=DEFAULT_SENS, align: str = "common") -> pd.DataFrame:
    """Per-iteration training table with columns reward, actor_loss, critic_loss, ex_time
    (+ shared_critic_loss for LACT_Off).

    PDMA     : ..._TOPO{n}_execution_log.csv, mean over servers.
    LACT_Off : ..._TOPO{n}training_log_server{b}.csv; per server the task terms (d = 0,1,2) are
               summed, then the result is averaged over servers.
               shared_critic_loss: mean over the shared_critic_loss-d columns, then over servers.
    """
    tier, n = tier_name(tier), n_servers(tier)
    d, tag = exp_dir(tier, exp.name), sens_tag(sens)

    if exp.family == "pdma":
        df = read_csv(locate(d, f"[{tag}]_{exp.name}_TOPO[{tier_to_idx(n)}]_execution_log.csv", exp.homo)[0])
        ids = [b for b in range(n) if f"server_{b}_reward" in df.columns]
        out = pd.DataFrame(index=df.index)
        for key in ("reward", "actor_loss", "critic_loss"):
            out[key] = df[[f"server_{b}_{key}" for b in ids]].astype(float).mean(axis=1)
        out["ex_time"] = df["ex_time"].astype(float) if "ex_time" in df.columns else np.nan
        return out

    frames = []
    for b in range(n):
        hits = locate(d, f"[{tag}]_{exp.name}_TOPO[{tier_to_idx(n)}]training_log_server{b}.csv", exp.homo, required=False)
        if not hits:
            continue
        df = read_csv(hits[0])

        def cols(prefix):
            return [c for c in df.columns if re.fullmatch(rf"{prefix}-\d+", c)]

        f = pd.DataFrame(index=df.index)
        f["reward"] = df[cols("reward")].astype(float).sum(axis=1, min_count=1)
        f["actor_loss"] = df[cols("actor_loss")].astype(float).sum(axis=1, min_count=1)
        f["critic_loss"] = df[cols("critic_loss")].astype(float).sum(axis=1, min_count=1)
        sc = [
            c for c in df.columns
            if "shared_critic_loss" in str(c).lower()
        ]
        f["shared_critic_loss"] = df[sc].astype(float).mean(axis=1) if sc else np.nan
        f["ex_time"] = df["ex_time"].astype(float) if "ex_time" in df.columns else np.nan
        frames.append(f)
    if not frames:
        raise FileNotFoundError(f"No LACT_Off server training logs found in {d}")
    return align_mean(frames, align)


# --------------------------------------------------------------------------- #
# Plot / CLI helpers
# --------------------------------------------------------------------------- #
def save_fig(fig, analysis_name: str, caption: str) -> Path:
    """Save as {analysis_name}_{caption}.png in cfg.GRAPH_DIR (non-alphanumerics -> '_')."""
    out_dir = Path(cfg.GRAPH_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = re.sub(r"[^A-Za-z0-9]+", "_", f"{analysis_name}_{caption}").strip("_") + ".png"
    path = out_dir / fname
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--tier", default="TIER_6", help="e.g. TIER_6 (default) or just 6")
    p.add_argument("--sens", nargs=3, type=int, default=list(DEFAULT_SENS), metavar=("C", "M", "S"),
                   help="sensitivity exponents c m s (default 0 0 0)")
    p.add_argument("--align", choices=["common", "union"], default="common",
                   help="how to merge files with different iteration counts (default: common)")
    p.add_argument("--smooth", type=int, default=1, help="moving-average window for line plots (default 1 = off)")
    return p
