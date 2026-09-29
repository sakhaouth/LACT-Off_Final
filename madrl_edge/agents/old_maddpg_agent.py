"""
Per-agent MADDPG wrapper + coordinator across all NUM_SERVERS agents.

Incorporates the fixes from prior debugging sessions:
  - target-critic soft update ONLY fires alongside a delayed actor update
    (previously this ran every critic step and caused divergence)
  - per-agent critic_update_count (previously a single shared counter caused
    only alternating agents to receive actor updates)
"""
import copy
import torch
import torch.nn.functional as F
import numpy as np

from madrl_edge import config
from madrl_edge.models.networks import Actor, Critic
import config as cfg



class MADDPGAgent:
    def __init__(self, agent_id, num_agents=cfg.NUM_SERVERS,
                 obs_dim=cfg.OBS_DIM, action_dim=cfg.MADDPG_ACTION_DIM, device=cfg.DEVICE):
        self.agent_id = agent_id
        self.device = device

        self.actor = Actor(obs_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(num_agents, obs_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.CRITIC_LR,
            weight_decay=config.CRITIC_WEIGHT_DECAY,
        )

        # per-agent counter -- do NOT share this across agents
        self.critic_update_count = 0

    def act(self, obs, migration_mask=None, noise_std=0.0):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = None
        if migration_mask is not None:
            mask_t = torch.as_tensor(migration_mask, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t, migration_mask=mask_t).squeeze(0).cpu().numpy()
        if noise_std > 0:
            # CONFUSION: exploration noise on a softmax/sigmoid-constrained
            # action needs care -- naive additive Gaussian noise can break
            # the sum-to-1 constraint on the allocation dims. Here we add
            # noise only pre-squash-equivalent by re-normalizing after; a
            # cleaner approach is Dirichlet-noise for the allocation head and
            # simple clipped Gaussian for the migration head.
            action = action + np.random.normal(0, noise_std, size=action.shape)
            action[0:6] = _fix_allocation(action[0:6])
            action[6:12] = np.clip(action[6:12], 0.0, 1.0)
        return action

    def soft_update(self, update_actor: bool):
        """
        Only soft-update targets when this call coincides with an actor
        update (TD3-style delay). This is the fix for the divergence bug
        where target_critic was updated every critic step regardless of
        update_actor, creating a positive feedback loop.
        """
        if not update_actor:
            return
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(config.TAU * p.data + (1 - config.TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(config.TAU * p.data + (1 - config.TAU) * tp.data)


def _fix_allocation(alloc6):
    cpu = alloc6[[0, 2, 4]]
    mem = alloc6[[1, 3, 5]]
    cpu = np.clip(cpu, 1e-6, None)
    mem = np.clip(mem, 1e-6, None)
    cpu = cpu / cpu.sum()
    mem = mem / mem.sum()
    out = np.empty(6, dtype=alloc6.dtype)
    out[[0, 2, 4]] = cpu
    out[[1, 3, 5]] = mem
    return out


class MADDPGCoordinator:
    """Owns all agents, does the joint update step."""

    def __init__(self, num_agents=cfg.NUM_SERVERS, device=cfg.DEVICE):
        self.num_agents = num_agents
        self.device = device
        self.agents = [MADDPGAgent(i, num_agents, device=device) for i in range(num_agents)]

    def act_all(self, obs_n, migration_masks=None, noise_std=0.0):
        masks = migration_masks or [None] * self.num_agents
        return [
            self.agents[i].act(obs_n[i], migration_mask=masks[i], noise_std=noise_std)
            for i in range(self.num_agents)
        ]

    def update(self, buffer, batch_size=config.BATCH_SIZE):
        if len(buffer) < max(batch_size, config.WARMUP_STEPS):
            return None

        obs_n, action_n, reward_n, next_obs_n, done_n = buffer.sample(batch_size)
        obs_n = torch.as_tensor(obs_n, device=self.device)             # (B,N,obs)
        action_n = torch.as_tensor(action_n, device=self.device)       # (B,N,act)
        reward_n = torch.as_tensor(reward_n, device=self.device)       # (B,N)
        next_obs_n = torch.as_tensor(next_obs_n, device=self.device)   # (B,N,obs)
        done_n = torch.as_tensor(done_n, device=self.device)           # (B,N)

        B, N, obs_dim = obs_n.shape
        act_dim = action_n.shape[-1]

        # target actions for next_obs, from each agent's TARGET actor
        with torch.no_grad():
            next_actions = torch.stack(
                [self.agents[i].target_actor(next_obs_n[:, i, :]) for i in range(N)],
                dim=1,
            )  # (B,N,act)

        flat_obs = obs_n.reshape(B, N * obs_dim)
        flat_next_obs = next_obs_n.reshape(B, N * obs_dim)
        flat_actions = action_n.reshape(B, N * act_dim)
        flat_next_actions = next_actions.reshape(B, N * act_dim)

        losses = {}
        for i, agent in enumerate(self.agents):
            # ---- critic update ----
            with torch.no_grad():
                target_q = agent.target_critic(flat_next_obs, flat_next_actions).squeeze(-1)
                y = reward_n[:, i] + config.GAMMA * (1 - done_n[:, i]) * target_q

            q = agent.critic(flat_obs, flat_actions).squeeze(-1)
            critic_loss = F.mse_loss(q, y)

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            agent.critic_update_count += 1
            update_actor = (agent.critic_update_count % config.ACTOR_UPDATE_DELAY == 0)

            actor_loss_val = None
            if update_actor:
                # ---- actor update: re-compute this agent's action with
                # current (non-target) actor, keep others fixed ----
                actions_for_actor_loss = action_n.clone()
                this_action = agent.actor(obs_n[:, i, :])
                actions_for_actor_loss = torch.cat(
                    [actions_for_actor_loss[:, :i, :], this_action.unsqueeze(1),
                     actions_for_actor_loss[:, i + 1:, :]],
                    dim=1,
                )
                flat_actions_for_actor = actions_for_actor_loss.reshape(B, N * act_dim)
                actor_loss = -agent.critic(flat_obs, flat_actions_for_actor).mean()

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()
                actor_loss_val = actor_loss.item()

            agent.soft_update(update_actor=update_actor)

            losses[i] = {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

        return losses
