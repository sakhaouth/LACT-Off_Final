"""
networks.py
===========
Actor / Critic / SharedCritic networks for the Multi-Task A2C-with-Shared-
Critic trainer.

Each server owns NUM_DECISIONS=3 (actor, critic) pairs -- one for
ALLOCATION, one for MIGRATION, one for SHARING -- plus a single
SharedCritic, matching "each agent will contain 3 actors and 3 critics
networks and 1 shared critic network". All three actors take the SAME
state (local + combined neighbor load) and output ACTION_DIM=6 values
(3 queues x {cpu, memory}).

CONFUSED: the prompt doesn't say whether the 6 outputs of a given decision
should sum to 1 (a genuine "proportion" split across the 3 queues, capped
by the server's total capacity) or should each be an independent [0,1]
fraction of that queue's own demand. I read ALLOCATION as the former (you
can't allocate more than 100% of the server's CPU/memory *in total*, so the
per-resource triplet should sum to 1) and MIGRATION/SHARING as the latter
(each queue independently decides what fraction of *itself* to
migrate/share, with no cross-queue coupling). `DECISION_ACTION_KIND` below
lets you flip this per decision type if that reading is wrong.

FIX (stability): `DecisionActor.forward()` previously only floored the
policy's std (`clamp(min=1e-3)`) with no ceiling. Because log-probs/entropy
are computed in *pre-squash* Gaussian space (see the class docstring below)
but the actual action is squashed through sigmoid/softmax, the entropy
bonus can keep rewarding a growing `log_std` even once the squash has
already saturated -- i.e. even after the entropy stops corresponding to
any real behavioral randomness in the *squashed* action. Nothing bounded
`log_std` from above, so it could drift upward without limit over training,
making the policy increasingly erratic/high-variance. `log_std` is now
clamped on both ends before exponentiating (a standard bound for
squashed-Gaussian policies, matching e.g. SAC's typical [-20, 2] convention,
tightened here to [-5, 2] since the backbone already starts near 0).
"""

import enum
import torch
import torch.nn as nn
from torch.distributions import Normal

import config as cfg


class ActionKind(enum.Enum):
    SOFTMAX_PER_RESOURCE = "softmax"  # ALLOCATION: proportions sum to 1 per resource
    SIGMOID_INDEPENDENT = "sigmoid"   # MIGRATION / SHARING: independent fractions


DECISION_ACTION_KIND = {
    cfg.ALLOCATION: ActionKind.SOFTMAX_PER_RESOURCE,
    cfg.MIGRATION: ActionKind.SIGMOID_INDEPENDENT,
    cfg.SHARING: ActionKind.SIGMOID_INDEPENDENT,
}

# Bounds for the actor's learned log-std, applied before exponentiating.
# Prevents the entropy bonus from driving std to infinity (upper bound) and
# prevents the policy from collapsing to a near-deterministic point mass too
# early (lower bound), both of which destabilize the policy gradient.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim, out_dim, hidden=cfg.HIDDEN_DIM):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class DecisionActor(nn.Module):
    """
    Stochastic policy for one decision type (allocation / migration /
    sharing). Outputs a Normal distribution over ACTION_DIM raw logits; the
    logits are squashed into a valid action (softmax-per-resource or
    independent sigmoids -- see `ActionKind`) by `to_action()`.

    CONFUSED / SIMPLIFICATION: log-probabilities used for the policy
    gradient are computed in the *pre-squash* Gaussian space, not corrected
    for the softmax/sigmoid change-of-variables (as an exactly-correct
    squashed-Gaussian policy would do with a log-det-Jacobian term). This
    is a common, reasonable simplification and works fine in practice; if
    an exactly-correct continuous policy gradient is required, add that
    correction term inside `log_prob_of()`. Precisely BECAUSE this
    correction is skipped, bounding `log_std` (see module docstring) matters
    more here than in an exact implementation -- it's the only thing
    keeping the pre-squash entropy bonus from diverging.
    """

    def __init__(self, action_kind: ActionKind, state_dim=cfg.A2C_STATE_DIM, action_dim=cfg.A2C_ACTION_DIM):
        super().__init__()
        self.action_kind = action_kind
        self.action_dim = action_dim
        self.backbone = _mlp(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state):
        mean = self.backbone(state)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mean, std)

    def to_action(self, raw_sample):
        """Squash a raw Gaussian sample into a valid action in [0, 1]^6."""
        if self.action_kind == ActionKind.SIGMOID_INDEPENDENT:
            return torch.sigmoid(raw_sample)
        # SOFTMAX_PER_RESOURCE: split into {cpu(3), mem(3)} and softmax each
        half = raw_sample.shape[-1] // 2
        cpu, mem = raw_sample[..., :half], raw_sample[..., half:]
        return torch.cat([torch.softmax(cpu, dim=-1), torch.softmax(mem, dim=-1)], dim=-1)

    def act(self, state):
        """Sample an action; return (env_action, raw_sample, log_prob, entropy)."""
        dist = self.forward(state)
        raw_sample = dist.rsample()
        log_prob = dist.log_prob(raw_sample).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        env_action = self.to_action(raw_sample)
        return env_action, raw_sample, log_prob, entropy

    def log_prob_of(self, state, raw_sample):
        dist = self.forward(state)
        return dist.log_prob(raw_sample).sum(dim=-1), dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    """Task-specific (local) critic: V(s) for one decision type."""

    def __init__(self, state_dim=cfg.A2C_STATE_DIM):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class SharedCritic(nn.Module):
    """
    One shared critic per server, used by all 3 decision types to compute
    the "shared advantage" A^S_i in Algorithm 2.
    """

    def __init__(self, state_dim=cfg.A2C_STATE_DIM):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)