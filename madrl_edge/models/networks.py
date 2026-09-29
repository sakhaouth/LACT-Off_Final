"""
Actor and Critic networks.

Actor enforces the two structural constraints from the spec architecturally
(rather than hoping the policy learns them):
  - allocation: cpu_high+cpu_mid+cpu_low == 1  -> softmax over 3 logits
  - allocation: mem_high+mem_mid+mem_low == 1  -> softmax over 3 logits
  - migration: each of the 6 fractions independently in [0,1] -> sigmoid
"""
import torch
import torch.nn as nn

# from madrl_edge import config
import config as cfg

class Actor(nn.Module):
    def __init__(self, obs_dim=cfg.OBS_DIM, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 3 logits for cpu allocation across HIGH/MID/LOW
        self.cpu_alloc_head = nn.Linear(hidden, cfg.NUM_QUEUES)
        # 3 logits for memory allocation across HIGH/MID/LOW
        self.mem_alloc_head = nn.Linear(hidden, cfg.NUM_QUEUES)
        # 6 raw values for migration fractions (per-queue cpu+mem), squashed via sigmoid
        self.migration_head = nn.Linear(hidden, cfg.NUM_QUEUES * 2)

    def forward(self, obs, migration_mask=None):
        """
        obs: (batch, OBS_DIM)
        migration_mask: optional (batch, NUM_QUEUES*2) of 0/1, used to zero out
            migration dimensions for servers with NO neighbours (e.g. the old
            BS3 empty-NEIGHBORS case) so the actor isn't exploring an
            infeasible action. CONFUSION: masking approach (multiply post-hoc
            vs. exclude from loss) is a design choice; here we just zero the
            output, which is simplest but means gradients for masked dims are
            near-zero, which is fine since masked servers should never migrate.
        Returns a 12-dim action vector per the layout in config.py.
        """
        h = self.body(obs)

        cpu_logits = self.cpu_alloc_head(h)
        mem_logits = self.mem_alloc_head(h)
        cpu_alloc = torch.softmax(cpu_logits, dim=-1)   # (batch, 3)
        mem_alloc = torch.softmax(mem_logits, dim=-1)   # (batch, 3)

        migration_raw = self.migration_head(h)
        migration_frac = torch.sigmoid(migration_raw)   # (batch, 6)
        if migration_mask is not None:
            migration_frac = migration_frac * migration_mask

        # interleave into the [cpu_h, mem_h, cpu_m, mem_m, cpu_l, mem_l] layout
        alloc = torch.stack(
            [cpu_alloc[:, 0], mem_alloc[:, 0],
             cpu_alloc[:, 1], mem_alloc[:, 1],
             cpu_alloc[:, 2], mem_alloc[:, 2]],
            dim=-1,
        )  # (batch, 6)

        action = torch.cat([alloc, migration_frac], dim=-1)  # (batch, 12)
        return action


class Critic(nn.Module):
    """
    Centralized critic: takes ALL agents' observations and ALL agents'
    actions (CTDE), outputs a single Q-value. One critic instance per agent
    (standard MADDPG), or a shared critic with agent-id conditioning --
    CONFUSION: spec doesn't say whether critics are per-agent or shared;
    implemented as per-agent (standard Lowe et al. 2017 MADDPG), matching
    your earlier maddpg.py foundation.
    """

    def __init__(self, num_agents=cfg.NUM_SERVERS,
                 obs_dim=cfg.OBS_DIM, action_dim=cfg.A2C_ACTION_DIM, hidden=256):
        super().__init__()
        total_obs = obs_dim * num_agents
        total_act = action_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(total_obs + total_act, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
