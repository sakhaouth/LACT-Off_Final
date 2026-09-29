"""
agent.py
========
ServerAgent bundles the 3 (actor, critic) pairs -- one per decision type --
plus the 1 shared critic that Algorithm 2 says every server ("agent")
should own ("each agent will contain 3 actors and 3 critics networks and 1
shared critic network").

FIX (stability): two additions on top of the original, both aimed at the
critic_loss spikes seen in the training log (single digits one iteration,
tens of thousands a few iterations later):

1. Target networks (`target_critics`, `target_shared_critic`): frozen,
   Polyak-averaged copies of each critic, used only to compute the
   bootstrap target V(s') in `buffer.compute_td_advantage`. Without these,
   a critic is trained toward a target built from its own live, constantly
   -changing prediction -- a moving-target loop that's a well-known source
   of this kind of oscillation. `soft_update_target()` /
   `soft_update_shared_target()` nudge the targets toward the live weights
   by a small `tau` after every optimizer step, so they track slowly
   instead of jumping with every noisy gradient step.

2. Running normalizers (`state_norm`, `reward_norms`): rescale states and
   rewards online so the critic's input and target scales stay stable
   across a rollout, instead of tracking raw load sums (which can reach the
   hundreds/thousands depending on queue backlog) and un-harmonized
   per-decision reward scales.

FIX (actor-loss stability): `actor_schedulers` linearly anneal each actor's
learning rate over training (see `step_schedulers()`). Vanilla
policy-gradient actor updates have no built-in trust region -- see
train.py's PPO-style clipped surrogate objective for the main fix -- but
annealing the LR on top of that further shrinks step size later in
training, which is when a constant LR would otherwise keep producing
updates of a size that never lets the logged actor_loss settle down.
"""

import copy

import torch

import config as cfg
from mec_marl.networks import DecisionActor, Critic, SharedCritic, DECISION_ACTION_KIND
from mec_marl.running_stats import RunningMeanStd

# Polyak-averaging coefficient for target-network updates. Kept as a local
# default (rather than requiring a config.py edit) via getattr fallback --
# set cfg.TARGET_TAU to override.
TARGET_TAU = cfg.TARGET_TAU


class ServerAgent:
    def __init__(self):
        self.actors = {d: DecisionActor(DECISION_ACTION_KIND[d]).to(cfg.DEVICE) for d in range(cfg.NUM_DECISIONS)}
        self.critics = {d: Critic().to(cfg.DEVICE) for d in range(cfg.NUM_DECISIONS)}
        self.shared_critic = SharedCritic().to(cfg.DEVICE)

        # Frozen target copies -- see module docstring. Never trained
        # directly; only ever nudged via soft_update_*().
        self.target_critics = {d: copy.deepcopy(c) for d, c in self.critics.items()}
        self.target_shared_critic = copy.deepcopy(self.shared_critic)
        for net in list(self.target_critics.values()) + [self.target_shared_critic]:
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_optims = {
            d: torch.optim.Adam(self.actors[d].parameters(), lr=cfg.A2C_ACTOR_LR) for d in range(cfg.NUM_DECISIONS)
        }
        self.critic_optims = {
            d: torch.optim.Adam(self.critics[d].parameters(), lr=cfg.A2C_CRITIC_LR) for d in range(cfg.NUM_DECISIONS)
        }
        self.shared_critic_optim = torch.optim.Adam(self.shared_critic.parameters(), lr=cfg.A2C_CRITIC_LR)

        self.state_norm = RunningMeanStd(shape=(cfg.A2C_STATE_DIM,))
        self.reward_norms = {d: RunningMeanStd(shape=()) for d in range(cfg.NUM_DECISIONS)}
        # FIX (actor-loss stability): linearly anneal each actor's LR from
        # 1.0x down to 0.05x over the course of training. Vanilla
        # policy-gradient updates have no built-in trust region, so a
        # constant LR keeps producing update steps of comparable size even
        # once the policy is close to a good behavior -- annealing shrinks
        # step size over time so late-training updates (and therefore the
        # logged actor_loss) settle down instead of oscillating forever.
        # Call `step_schedulers()` once per outer training iteration (see
        # train.py main()).
        _num_iters = max(getattr(cfg, "NUM_ITERATIONS", 100), 1)
        self.actor_schedulers = {
            d: torch.optim.lr_scheduler.LambdaLR(
                self.actor_optims[d],
                lr_lambda=lambda it, _T=_num_iters: max(1.0 - it / _T, 0.05),
            )
            for d in range(cfg.NUM_DECISIONS)
        }

    def step_schedulers(self):
        """Advance all actor LR schedules by one outer training iteration."""
        for sched in self.actor_schedulers.values():
            sched.step()

        # This server's state normalizer (shared across all 3 decisions --
        # they all consume the same STATE_DIM local+neighbor-load vector),
        # and one reward normalizer per decision type (allocation /
        # migration / sharing rewards have different natural scales, see
        # environment.py, so they're tracked separately).
        self.state_norm = RunningMeanStd(shape=(cfg.STATE_DIM,))
        self.reward_norms = {d: RunningMeanStd(shape=()) for d in range(cfg.NUM_DECISIONS)}

    def act(self, decision, state_tensor):
        return self.actors[decision].act(state_tensor)

    def normalize_state(self, state_np, update: bool = True):
        """Normalize a raw state vector with this server's running
        mean/std. `update=True` (the default) keeps the running estimate
        adapting online as new slots are observed."""
        return self.state_norm.normalize(state_np, center=True, update=update)

    def normalize_reward(self, decision, reward, update: bool = True):
        """Scale (but do not re-center) a raw reward for `decision` using a
        running std estimate. Centering is deliberately skipped: 0 reward
        already means "nothing completed or dropped this slot," a
        meaningful reference point that mean-subtraction would erase."""
        return float(self.reward_norms[decision].normalize(reward, center=False, update=update))

    def soft_update_target(self, decision, tau: float = TARGET_TAU):
        self._polyak(self.target_critics[decision], self.critics[decision], tau)

    def soft_update_shared_target(self, tau: float = TARGET_TAU):
        self._polyak(self.target_shared_critic, self.shared_critic, tau)

    @staticmethod
    def _polyak(target_net, source_net, tau: float):
        with torch.no_grad():
            for tp, sp in zip(target_net.parameters(), source_net.parameters()):
                tp.mul_(1.0 - tau).add_(sp, alpha=tau)

    def state_dict(self):
        return {
            "actors": {d: a.state_dict() for d, a in self.actors.items()},
            "critics": {d: c.state_dict() for d, c in self.critics.items()},
            "shared_critic": self.shared_critic.state_dict(),
            "target_critics": {d: c.state_dict() for d, c in self.target_critics.items()},
            "target_shared_critic": self.target_shared_critic.state_dict(),
        }

    def load_state_dict(self, sd):
        for d, a in self.actors.items():
            a.load_state_dict(sd["actors"][d])
        for d, c in self.critics.items():
            c.load_state_dict(sd["critics"][d])
        self.shared_critic.load_state_dict(sd["shared_critic"])
        if "target_critics" in sd:
            for d, c in self.target_critics.items():
                c.load_state_dict(sd["target_critics"][d])
            self.target_shared_critic.load_state_dict(sd["target_shared_critic"])
        else:
            # Loading an old checkpoint without target nets -- initialize
            # targets from the freshly-loaded live weights.
            for d, c in self.target_critics.items():
                c.load_state_dict(self.critics[d].state_dict())
            self.target_shared_critic.load_state_dict(self.shared_critic.state_dict())