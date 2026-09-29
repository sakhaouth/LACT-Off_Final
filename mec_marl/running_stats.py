"""
running_stats.py
=================
Online running-mean/variance tracker (Welford-style incremental update),
used to normalize states and rewards on the fly during training.

WHY THIS EXISTS: the training log showed critic_loss spiking into the tens
of thousands (e.g. 296565 at iter 0, repeated spikes to 8k-30k later) while
reward stayed in a tight [-2.7, -0.1] band. That mismatch points at two
compounding causes: (1) unnormalized state inputs -- local_load sums raw
cpu/memory/tolerance/count across every waiting task, which can reach the
hundreds or thousands depending on queue backlog, so the same network
sees wildly different input scales slot to slot -- and (2) the different
reward streams (allocation / migration / sharing) not being on comparable
scales with each other or over time. Both make the value target's scale
swing around, which is what makes MSE-style critic loss spike.

`RunningMeanStd` fixes this by tracking a running estimate of mean/std for
whatever it's given and rescaling accordingly, adapting online as new data
arrives (no separate calibration pass needed).
"""

import numpy as np


class RunningMeanStd:
    """Tracks running mean/variance for scalars or fixed-size vectors."""

    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        # Incremental (biased) variance estimate -- adapts online rather
        # than requiring a fixed pre-collected batch, which matters here
        # since the task-arrival distribution and reward scale can drift
        # over a long training run.
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x, center: bool = True, clip: float = 10.0, update: bool = True):
        """Normalize `x` using the current running stats.

        center=True subtracts the mean too (appropriate for states, where
        the raw scale/offset carries no special meaning). center=False only
        divides by std (appropriate for rewards, where 0 already has a
        meaningful interpretation -- e.g. "nothing completed or dropped
        this slot" -- that mean-subtraction would erase).
        """
        x_np = np.asarray(x, dtype=np.float64)
        if update:
            self.update(x_np)
        std = np.sqrt(np.maximum(self.var, 1e-8))
        if center:
            out = (x_np - self.mean) / (std + 1e-6)
        else:
            out = x_np / (std + 1e-6)
        out = np.clip(out, -clip, clip)
        return out.astype(np.float32)