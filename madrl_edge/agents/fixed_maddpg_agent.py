"""
Per-agent MADDPG wrapper + coordinator across all NUM_SERVERS agents.

Carries forward the earlier fixes:
  - target-critic soft update ONLY fires alongside a delayed actor update
  - per-agent critic_update_count
  - RunningNorm reward normalization + clipping
  - gradient clipping on actor and critic
  - NaN/Inf batch-skip guards
  - twin critics (critic1/critic2, target_critic1/target_critic2) per agent
  - TD target uses min(target_critic1, target_critic2)
  - actor trained only against critic1 (TD3)
  - target policy smoothing with projection back onto the action manifold
  - belt-and-braces TD-target clamp (Q_VALUE_CLIP)

REVISION (this pass): despite all of the above, the 200-episode log still
shows critic_loss and actor_loss growing/oscillating instead of converging
(server_0 is the worst case: critic_loss climbs from ~200 to multi-thousand
territory and never settles, while actor_loss swings from -1900 to +2900
across the run instead of trending toward zero). Two concrete bugs were
found that explain this on top of the fixes already in place:

1. DOUBLE REWARD NORMALIZATION. fixed_train.py normalizes reward_n with its
   own running normalizer BEFORE pushing it into the replay buffer. This
   file's RunningNorm then normalizes it AGAIN when a batch is sampled back
   out in `update()`. Two problems compound here:
     - the same transition gets a *different* effective scale depending on
       when it was pushed (train.py's stats at push-time) vs. when it's
       sampled (this file's stats at sample-time, computed over whatever
       ages happen to land in that minibatch) -- the TD target for the
       exact same stored transition silently drifts over the course of
       training, which is a moving-target problem layered on top of the
       normal (and already-handled) non-stationarity of MADDPG.
     - normalizing an already fairly-standardized value again amplifies
       small estimation noise in the second normalizer's running stats.
   Fix: reward normalization now happens in exactly ONE place -- here, at
   sample time, which is also where it's actually needed (right before it
   enters the TD target). fixed_train.py has been changed to push the RAW
   env reward into the buffer instead.

2. ACTOR/TARGET-ACTOR CALLED WITHOUT migration_mask DURING update(). At
   rollout time (`act()`), the actor receives a `migration_mask` describing
   whether this server has any neighbors to migrate to. During training,
   both the target-action computation and the actor-loss recomputation
   called `agent.actor(...)` / `agent.target_actor(...)` with NO mask at
   all -- i.e. the actor was trained on an input distribution it never
   actually sees at inference time. Since `migration_mask` only depends on
   the (fixed) topology, not on runtime state, it's the same constant
   vector for a given agent for the entire run -- so it's stored once per
   agent here and now passed consistently everywhere the actor/target actor
   is invoked.
"""
import copy
import itertools
import warnings

import torch
import torch.nn.functional as F
import numpy as np

# from madrl_edge import config
from madrl_edge.models.networks import Actor, Critic
import config as cfg

# ---------------------------------------------------------------------------
# Target policy smoothing / TD-target clamp constants.
# Pulled from config.py if present, otherwise sensible defaults so this file
# doesn't hard-require a config.py edit to run.
# ---------------------------------------------------------------------------
TARGET_POLICY_NOISE = getattr(cfg, "TARGET_POLICY_NOISE", 0.1)
TARGET_NOISE_CLIP = getattr(cfg, "TARGET_NOISE_CLIP", 0.3)
# If not set explicitly, derive a bound from the reward-normalization clip
# and the discount factor: with a normalized/clipped reward in
# [-REWARD_NORM_CLIP, REWARD_NORM_CLIP], the fixed point of the Bellman
# recursion is bounded by REWARD_NORM_CLIP / (1 - GAMMA). We give a little
# headroom (1.5x) since bootstrapped estimates overshoot during learning.
# NOTE: if GAMMA is very close to 1 this bound gets very loose (e.g.
# GAMMA=0.99 -> 150x the reward clip) and stops acting like a real
# constraint well before the critic has converged. If critic_loss keeps
# growing toward this ceiling after the two fixes below, that's the next
# thing to tighten (either lower GAMMA for this task's effective horizon,
# or set Q_VALUE_CLIP explicitly in config.py instead of relying on the
# derived default).
_DEFAULT_Q_CLIP = 1.5 * getattr(cfg, "REWARD_NORM_CLIP", 10.0) / max(1e-6, (1 - cfg.GAMMA))
Q_VALUE_CLIP = getattr(cfg, "Q_VALUE_CLIP", _DEFAULT_Q_CLIP)


class RunningNorm:
    """Welford-style running mean/variance for reward normalization.

    This is now the ONLY place reward normalization happens (see module
    docstring, fix #1) -- fixed_train.py stores the raw env reward in the
    replay buffer, and this class normalizes it here, once, right before it
    enters the TD target.
    """

    def __init__(self, eps=cfg.REWARD_NORM_EPS):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.eps = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        batch_count = x.size
        if batch_count == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x, clip=cfg.REWARD_NORM_CLIP):
        std = (self.var + self.eps) ** 0.5
        normed = (x - self.mean) / std
        return torch.clamp(normed, -clip, clip)


def _project_action(action: torch.Tensor) -> torch.Tensor:
    """
    Re-project a (possibly noised) 12-dim action back onto the actor's
    structured manifold: [cpu_h, mem_h, cpu_m, mem_m, cpu_l, mem_l,
    mig_h_cpu, mig_h_mem, mig_m_cpu, mig_m_mem, mig_l_cpu, mig_l_mem].

    Needed for TD3-style target policy smoothing: additive Gaussian noise on
    a softmax-simplex output can push individual components negative or
    break the sum-to-1 constraint the Actor architecturally enforces, so we
    clamp to non-negative and renormalize each 3-way allocation block, and
    independently clip the 6 migration fractions to [0, 1].
    """
    cpu_idx = [0, 2, 4]
    mem_idx = [1, 3, 5]
    out = action.clone()

    cpu = out[..., cpu_idx].clamp_min(1e-6)
    cpu = cpu / cpu.sum(dim=-1, keepdim=True)
    out[..., cpu_idx] = cpu

    mem = out[..., mem_idx].clamp_min(1e-6)
    mem = mem / mem.sum(dim=-1, keepdim=True)
    out[..., mem_idx] = mem

    out[..., 6:] = out[..., 6:].clamp(0.0, 1.0)
    return out


class MADDPGAgent:
    def __init__(self, agent_id, num_agents=cfg.NUM_SERVERS,
                 obs_dim=cfg.OBS_DIM, action_dim=cfg.MADDPG_ACTION_DIM, device=cfg.DEVICE):
        self.agent_id = agent_id
        self.device = device

        self.actor = Actor(obs_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)

        # twin critics (TD3-style clipped double-Q). Two independently
        # initialized Critic nets -- min(Q1, Q2) removes the single-critic
        # overestimation bias.
        self.critic1 = Critic(num_agents, obs_dim, action_dim).to(device)
        self.critic2 = Critic(num_agents, obs_dim, action_dim).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR)
        self.critic_opt = torch.optim.Adam(
            itertools.chain(self.critic1.parameters(), self.critic2.parameters()),
            lr=cfg.CRITIC_LR,
            weight_decay=cfg.CRITIC_WEIGHT_DECAY,
        )

        # per-agent counter -- do NOT share this across agents
        self.critic_update_count = 0

        # FIX (bug #2, see module docstring): migration_mask depends only
        # on this agent's fixed topology (does it have any neighbors to
        # migrate to?), exactly mirroring BaseStation.migration_mask() in
        # environment.py. It's therefore a constant for the agent's whole
        # lifetime, so we compute and cache it here instead of leaving
        # update() to call the actor/target_actor with no mask at all.
        has_neighbors = len(cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][agent_id]) > 0
        mask_val = 1.0 if has_neighbors else 0.0
        self.migration_mask = torch.full(
            (cfg.NUM_QUEUES * 2,), mask_val, dtype=torch.float32, device=device
        )

    def act(self, obs, migration_mask=None, noise_std=0.0):
        obs_arr = np.asarray(obs, dtype=np.float64)
        if not np.isfinite(obs_arr).all():
            # FIX: last line of defense. If a NaN/Inf slipped into obs from
            # the load predictor or the frozen VRNN encoder (both sit
            # upstream of the actor and are the actual sources seen in
            # practice), sanitize it HERE so a corrupted action can never be
            # written into the replay buffer in the first place. The
            # upstream guards in environment.py should catch this earlier;
            # this is a backstop, not a substitute for fixing the source.
            warnings.warn(
                f"MADDPGAgent {self.agent_id}: non-finite obs passed to act(), "
                "sanitizing to zero before running the actor."
            )
            obs = np.nan_to_num(obs_arr, nan=0.0, posinf=0.0, neginf=0.0)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = None
        if migration_mask is not None:
            mask_t = torch.as_tensor(migration_mask, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t, migration_mask=mask_t).squeeze(0).cpu().numpy()
        if not np.isfinite(action).all():
            warnings.warn(
                f"MADDPGAgent {self.agent_id}: non-finite action produced by "
                "actor even with finite input; sanitizing before use."
            )
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        if noise_std > 0:
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
            tp.data.copy_(cfg.TAU * p.data + (1 - cfg.TAU) * tp.data)
        for p, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.copy_(cfg.TAU * p.data + (1 - cfg.TAU) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.copy_(cfg.TAU * p.data + (1 - cfg.TAU) * tp.data)


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
        # one running normalizer per agent -- reward scale can differ a lot
        # between servers (different capacities / backlog levels). This is
        # now the ONLY place reward normalization happens (see fix #1).
        self.reward_norms = [RunningNorm() for _ in range(num_agents)]
        self.q_clip = Q_VALUE_CLIP

    def act_all(self, obs_n, migration_masks=None, noise_std=0.0):
        masks = migration_masks or [None] * self.num_agents
        return [
            self.agents[i].act(obs_n[i], migration_mask=masks[i], noise_std=noise_std)
            for i in range(self.num_agents)
        ]

    def update(self, buffer, batch_size=cfg.MADDPG_BATCH_SIZE):
        if len(buffer) < max(batch_size, cfg.WARMUP_STEPS):
            return None

        # NOTE: reward_n sampled here is now the RAW env reward -- see
        # fixed_train.py, which no longer normalizes before pushing into the
        # buffer. Normalization happens once, below.
        obs_n, action_n, reward_n, next_obs_n, done_n = buffer.sample(batch_size)

        # FIX: a NaN/Inf anywhere in the sampled batch means a previous,
        # already-corrupted transition is in the buffer. Skip this update
        # rather than feed it through the network and spread the corruption
        # further -- previously nothing guarded against this.
        if not (np.isfinite(obs_n).all() and np.isfinite(action_n).all()
                and np.isfinite(reward_n).all() and np.isfinite(next_obs_n).all()):
            warnings.warn("MADDPGCoordinator.update: non-finite values in sampled "
                          "batch, skipping this update step.")
            return None

        obs_n = torch.as_tensor(obs_n, device=self.device)             # (B,N,obs)
        action_n = torch.as_tensor(action_n, device=self.device)       # (B,N,act)
        reward_n = torch.as_tensor(reward_n, device=self.device)       # (B,N)
        next_obs_n = torch.as_tensor(next_obs_n, device=self.device)   # (B,N,obs)
        done_n = torch.as_tensor(done_n, device=self.device)           # (B,N)

        B, N, obs_dim = obs_n.shape
        act_dim = action_n.shape[-1]

        # FIX (bug #1, RE-APPLIED): normalize each agent's RAW reward with a
        # running mean/std -- exactly once, here -- before it enters the TD
        # target. Raw rewards scale with unbounded queue backlog
        # (VIOLATION_PENALTY summed over every dropped task, plus
        # BACKLOG_PENALTY_WEIGHT * backlog_len), so an un-normalized TD
        # target drifts as the policy's drop/backlog behaviour changes over
        # training -- this is what was actually causing the reward curve to
        # suddenly drop and the critic loss to fail to converge: the
        # Q_VALUE_CLIP bound derived above assumes REWARD_NORM_CLIP-scale
        # rewards are actually flowing into the target, but this call was
        # previously commented out (`norm_reward_n = reward_n`, i.e. a
        # no-op), so raw reward -- not the assumed-clipped one -- was
        # feeding straight into the TD target while the clamp below was
        # calibrated for a distribution the reward never actually had.
        #
        # This is done ONLY here (sample time, right before the TD target is
        # built) -- fixed_train.py pushes the RAW env reward into the buffer
        # and must NOT normalize before storage, or this would silently
        # become a double-normalization bug again (see module docstring).
        for i in range(N):
            self.reward_norms[i].update(reward_n[:, i].detach().cpu().numpy())
        norm_reward_n = torch.stack(
            [self.reward_norms[i].normalize(reward_n[:, i]) for i in range(N)],
            dim=1,
        )
        # target actions for next_obs, from each agent's TARGET actor, with
        # TD3-style target policy smoothing.
        with torch.no_grad():
            # FIX (bug #2): pass each agent's (static, per-topology)
            # migration_mask to target_actor -- previously called with no
            # mask at all, so the target action was computed on an input
            # distribution the actor never sees at rollout time.
            next_actions = torch.stack(
                [
                    self.agents[i].target_actor(
                        next_obs_n[:, i, :],
                        migration_mask=self.agents[i].migration_mask.unsqueeze(0).expand(B, -1),
                    )
                    for i in range(N)
                ],
                dim=1,
            )  # (B,N,act)

            # FIX: target policy smoothing. Small clipped noise on the
            # target action makes it harder for the critic to overfit to a
            # narrow, sharp Q-peak around one specific next-action -- this
            # is the second half of the TD3 fix (the first half is the twin
            # critic below). Raw additive noise would break the actor's
            # softmax-simplex / sigmoid structure, so we project back onto
            # that manifold afterwards.
            noise = torch.randn_like(next_actions) * TARGET_POLICY_NOISE
            noise = torch.clamp(noise, -TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            next_actions = _project_action(next_actions + noise)

        flat_obs = obs_n.reshape(B, N * obs_dim)
        flat_next_obs = next_obs_n.reshape(B, N * obs_dim)
        flat_actions = action_n.reshape(B, N * act_dim)
        flat_next_actions = next_actions.reshape(B, N * act_dim)

        losses = {}
        for i, agent in enumerate(self.agents):
            # ---- critic update ----
            with torch.no_grad():
                # FIX: clipped double-Q. Taking the min of two independently
                # trained target critics removes the systematic upward bias
                # a single critic has (it's trained on maximized actions, so
                # its errors skew optimistic) -- this is what was actually
                # missing despite being described as already fixed.
                tq1 = agent.target_critic1(flat_next_obs, flat_next_actions).squeeze(-1)
                tq2 = agent.target_critic2(flat_next_obs, flat_next_actions).squeeze(-1)
                target_q = torch.min(tq1, tq2)
                # FIX: belt-and-braces clamp. Even with twin critics, cap the
                # bootstrapped value at a level consistent with the
                # normalized/clipped reward scale, so a transient bad
                # estimate can't compound into the runaway plateaus seen in
                # the log (server_1/3 critic_loss stuck at ~20k+).
                target_q = torch.clamp(target_q, -self.q_clip, self.q_clip)
                y = norm_reward_n[:, i] + cfg.GAMMA * (1 - done_n[:, i]) * target_q
                y = torch.clamp(y, -self.q_clip, self.q_clip)

            q1 = agent.critic1(flat_obs, flat_actions).squeeze(-1)
            q2 = agent.critic2(flat_obs, flat_actions).squeeze(-1)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            # FIX: guard against a corrupted forward pass before touching
            # any weights. Previously an occasional NaN/Inf critic_loss was
            # backpropagated anyway, permanently poisoning that agent's
            # critic (and, via the shared flat_obs/flat_actions batch, every
            # other agent's critic on the next update call).
            if not torch.isfinite(critic_loss):
                warnings.warn(f"MADDPGCoordinator.update: non-finite critic_loss "
                              f"for agent {i}, skipping this agent's update.")
                losses[i] = {"critic_loss": None, "actor_loss": None}
                continue

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(agent.critic1.parameters(), agent.critic2.parameters()),
                cfg.MAX_GRAD_NORM,
            )
            agent.critic_opt.step()

            agent.critic_update_count += 1
            update_actor = (agent.critic_update_count % cfg.ACTOR_UPDATE_DELAY == 0)

            actor_loss_val = None
            if update_actor:
                # ---- actor update: re-compute this agent's action with
                # current (non-target) actor, keep others fixed ----
                actions_for_actor_loss = action_n.clone()
                # FIX (bug #2): pass this agent's migration_mask here too --
                # previously called as agent.actor(obs_n[:, i, :]) with no
                # mask, same train/inference mismatch as the target-actor
                # call above.
                this_action = agent.actor(
                    obs_n[:, i, :],
                    migration_mask=agent.migration_mask.unsqueeze(0).expand(B, -1),
                )
                actions_for_actor_loss = torch.cat(
                    [actions_for_actor_loss[:, :i, :], this_action.unsqueeze(1),
                     actions_for_actor_loss[:, i + 1:, :]],
                    dim=1,
                )
                flat_actions_for_actor = actions_for_actor_loss.reshape(B, N * act_dim)
                # FIX: actor is trained against critic1 ONLY (TD3), never
                # critic2 and never min(critic1, critic2). This keeps the
                # policy gradient from being pulled toward whichever critic
                # happens to be more overoptimistic on a given batch.
                actor_loss = -agent.critic1(flat_obs, flat_actions_for_actor).mean()

                if torch.isfinite(actor_loss):
                    agent.actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.MAX_GRAD_NORM)
                    agent.actor_opt.step()
                    actor_loss_val = actor_loss.item()
                    update_actor = True
                else:
                    warnings.warn(f"MADDPGCoordinator.update: non-finite actor_loss "
                                  f"for agent {i}, skipping actor step.")
                    update_actor = False

            agent.soft_update(update_actor=update_actor)

            losses[i] = {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

        return losses