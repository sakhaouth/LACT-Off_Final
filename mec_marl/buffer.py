"""
buffer.py
=========
Per-(server, decision) rollout storage (`RolloutBuffer`) and the one-step
TD-advantage / TD-loss computation (`compute_td_advantage`) used by both
the local and shared critics in Algorithm 2.

This file wasn't available when auditing the training log, so it's
reconstructed here to match the exact interface `train.py` already calls:
  - `RolloutBuffer.add(state, raw_sample, log_prob, entropy, reward, next_state)`
  - `RolloutBuffer.as_tensors()` -> (states, raw_samples, log_probs, entropies, rewards, next_states)
  - `compute_td_advantage(critic, states, rewards, next_states)` -> (advantage, critic_loss)

FIX (stability), see training-log analysis: reward stayed in a tight band
([-2.7, -0.1]) while critic_loss spiked into the tens of thousands
repeatedly. Two things in a naive TD implementation directly cause that:

1. Bootstrapping `V(s')` from the SAME live critic being trained (no target
   network) creates a moving-target feedback loop: every update to the
   critic changes the target for the next update, which can chase itself
   into large oscillations, especially with CRITIC_LR=1e-3.
   -> `compute_td_advantage` now accepts an optional `target_critic`
   (a Polyak-averaged, no-grad copy -- see `agent.py`) used ONLY to compute
   `V(s')`. If none is passed, falls back to the live critic (still
   correct, just less stable) so the function remains usable standalone.

2. Plain MSE squares the raw TD error, so a single unusually large
   transition dominates the whole batch's gradient. Switched to Huber loss
   (`smooth_l1_loss`): quadratic near zero, linear beyond it, so outliers
   still contribute signal without a single one blowing up the update.
"""

import torch
import torch.nn.functional as F

import config as cfg


class RolloutBuffer:
    """Stores one worker's (state, action, reward, next_state, ...)
    transitions for a single (server, decision) key across a rollout."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.raw_samples = []
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.next_states = []

    def add(self, state, raw_sample, log_prob, entropy, reward, next_state):
        self.states.append(torch.as_tensor(state, dtype=torch.float32, device = cfg.DEVICE))
        self.raw_samples.append(torch.as_tensor(raw_sample, dtype=torch.float32, device = cfg.DEVICE))
        self.log_probs.append(torch.as_tensor(log_prob, dtype=torch.float32, device = cfg.DEVICE))
        self.entropies.append(torch.as_tensor(entropy, dtype=torch.float32, device = cfg.DEVICE))
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float32, device = cfg.DEVICE))
        self.next_states.append(torch.as_tensor(next_state, dtype=torch.float32, device = cfg.DEVICE))

    def as_tensors(self):
        states = torch.stack(self.states)
        raw_samples = torch.stack(self.raw_samples)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        return states, raw_samples, log_probs, entropies, rewards, next_states

    def __len__(self):
        return len(self.states)


def compute_td_advantage(critic, states, rewards, next_states, gamma=cfg.A2C_GAMMA, target_critic=None):
    """
    One-step TD advantage + critic loss: A(s,a) = r + gamma*V(s') - V(s).

    `target_critic`, if given, is a frozen/Polyak-averaged copy of `critic`
    used only to evaluate V(s') for the bootstrap target -- this decouples
    "what we're updating" from "what we're updating towards" and is the
    main fix for the oscillating critic_loss seen in training. Gradients
    never flow into the target net (no_grad) or back into the actor
    (advantage is detached).
    """
    target_net = target_critic if target_critic is not None else critic

    values = critic(states)
    with torch.no_grad():
        next_values = target_net(next_states)
        targets = rewards + gamma * next_values

    advantage = (targets - values).detach()
    critic_loss = F.smooth_l1_loss(values, targets)
    return advantage, critic_loss