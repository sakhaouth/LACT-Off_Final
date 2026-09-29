"""
environment.py
===============
Runs ONE parallel "worker" (Algorithm 2's Nw) -- a full set of NUM_SERVERS
edge servers -- through ROLLOUT_LEN time slots, producing per-
(server, decision) transitions.

CONFUSED / DESIGN CHOICE: Algorithm 2's pseudocode nests "foreach task i ...
foreach worker w ... rollout T steps" as if each of the 3 decision types
got its OWN independent rollout. In this domain the 3 decisions
(allocation/migration/sharing) all happen inside the SAME time slot and
depend on each other's outcomes (e.g. migration changes what allocation
sees next), so this runs ONE environment rollout per worker and records
each decision type's own (state, action, reward, next_state) sub-
trajectory from that single rollout, rather than re-running the whole
environment three separate times. This still gives Algorithm 2 exactly
what it needs per task (a per-worker trajectory to compute advantages and
gradients from) -- it's just collected more efficiently and keeps the
physics consistent across decision types within a slot.

FIX (stability): raw states/rewards are now passed through each server's
agent-owned running normalizer (`agent.normalize_state` /
`agent.normalize_reward`, see agent.py) before being fed to the actor/
critic networks or stored in the rollout buffer. Previously, raw local-load
sums (which scale with queue backlog and can reach the hundreds/thousands)
and un-harmonized per-decision reward scales were fed straight into the
networks, contributing to the critic-loss spikes seen in training. Since
workers share the same `agents` dict (see `Worker.__init__`), every
worker's rollout updates and benefits from the same running statistics.
"""

import torch

import config as cfg
from mec_marl.edge_server import EdgeServer
from mec_marl.buffer import RolloutBuffer
from load_predictor.llm_inference import load_get_model, llm_inference
from load_predictor.lstm_inference import lstm_load_get_model, lstm_inference
import numpy as np

class Worker:
    def __init__(self, agents, id):
        """
        agents: dict {server_id: ServerAgent}, the SAME objects across all
        workers so gradients computed from any worker's rollout apply to
        the same underlying parameters -- matching "Nw workers per task"
        collecting data for one shared policy per task per server. This
        also means all workers share the same per-server running
        normalizers, so state/reward statistics are pooled across workers
        rather than tracked independently per worker.
        """
        self.id = id
        self.agents = agents
        self.servers = []
        for s in range(cfg.NUM_SERVERS):
            server = EdgeServer(s)
            # model, scaler, device = lstm_load_get_model(s)
            # server.set_load_predictor(model, scaler, device, lstm_inference)
            # model, scaler, device = load_get_model(s)
            # server.set_load_predictor(model, scaler, device, llm_inference)
            self.servers.append(server)
        self.buffers = {
            (s, d): RolloutBuffer() for s in range(cfg.NUM_SERVERS) for d in range(cfg.NUM_DECISIONS)
        }

    def reset(self):
        for s in self.servers:
            s.reset()
        for buf in self.buffers.values():
            buf.clear()

    def rollout(self, length=cfg.ROLLOUT_LEN):
        total_info = np.zeros(cfg.NUM_SERVERS * 4)
        cons_time = np.zeros(cfg.NUM_SERVERS * 3)
        for _ in range(length):
            val,con = self._step()
            # print("kkkkkkkkkkkkkkkkk")
            # print(val)
            # print(con)
            total_info += np.array(val)
            cons_time += np.array(con)
        cons_time = cons_time / length
        return total_info.tolist(), cons_time.tolist()

    def get_info(self):
        return {s.id: s.get_info() for s in self.servers}

    def _step(self):
        state_mailboxes = {s.id: {} for s in self.servers}
        resource_mailboxes = {s.id: {} for s in self.servers}
        reward_mailboxes = {s.id: {} for s in self.servers}

        # 1) local load generation + sharing
        for s in self.servers:
            s.local_load_generation()
        # print(f"load generation done of worker {self.id}")
        for s in self.servers:
            s.local_load_sharing(state_mailboxes)

        # 2) aggregate neighbor loads into this slot's state
        # print("after sharing load", state_mailboxes)
        for s in self.servers:
            s.load_aggregation(state_mailboxes)

        state_tensors = {
            s.id: torch.as_tensor(self.agents[s.id].normalize_state(s.get_state()), dtype = torch.float32, device = cfg.DEVICE)
            for s in self.servers
        }

        # 3) sample actions for all 3 decisions, all servers
        actions, raw_samples, log_probs, entropies = {}, {}, {}, {}
        for s in self.servers:
            agent = self.agents[s.id]
            for d in range(cfg.NUM_DECISIONS):
                a, raw, lp, ent = agent.act(d, state_tensors[s.id])
                actions[(s.id, d)] = a.detach().cpu().numpy()
                raw_samples[(s.id, d)] = raw.detach()
                log_probs[(s.id, d)] = lp.detach()

                entropies[(s.id, d)] = ent.detach()

        # 4) publish + collect sharing decisions
        for s in self.servers:
            s.publish_sharing_decision(actions[(s.id, cfg.SHARING)], resource_mailboxes)
        for s in self.servers:
            s.collect_sharing_offers(resource_mailboxes)
            s.compute_migration_likelihood()

        # 5) migrate tasks based on the migration decision + likelihood
        server_by_id = {s.id: s for s in self.servers}
        for s in self.servers:
            s.migrate_tasks(actions[(s.id, cfg.MIGRATION)], server_by_id)


        for s in self.servers:
            rasidulas = s.execute(actions[(s.id, cfg.ALLOCATION)])
        

        reward_components = {s.id: s.compute_reward_components() for s in self.servers}

        for s in self.servers:
            neighbour_rewards = {n: 0 for n in cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][s.id]}
            for n in cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][s.id]:
                neighbour_rewards[n] = reward_components[s.id]["success_term"]["sharing_tasks"][cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][s.id].index(n)] + reward_components[s.id]["violation_term"]["sharing_tasks"][cfg.CONFIG[cfg.CURRENT_TOPOLOGY]["NEIGHBORS"][s.id].index(n)]
            # print(neighbour_rewards, "jjjjjjjjjjj ", s.id)
            s.publish_reward(neighbour_rewards, reward_mailboxes)
        
        # 6) local execution + reward
        rewards = {}
        infos = []
        con_time = []

        #share the migration reward

        

        # print("Here I am waiting")
        for s in self.servers:
            # residuals = s.execute(actions[(s.id, cfg.ALLOCATION)])
            # total_reward = s.reward(residuals)
            # print("hhhhhhhhhhhhhhhhhhh-----------------------------------------------------------------------------------------hh")
            allocation_reward = reward_components[s.id]["success_term"]["allocation_tasks"] + reward_components[s.id]["violation_term"]["allocation_tasks"]
            migration_reward = sum(reward_mailboxes[s.id].values())
            sharing_reward = np.array(reward_components[s.id]["success_term"]["sharing_tasks"]).sum() + np.array( reward_components[s.id]["violation_term"]["sharing_tasks"]).sum()

            # ASSUMPTION: the single slot reward is shared identically by
            # all 3 decision-type learners for this server (mirrors the
            # prompt's own "reward sharing" step -- one outcome, jointly
            # caused by all 3 decisions, attributed to all 3 learners). An
            # alternative would decompose the reward per decision type
            # (e.g. an explicit migration-cost penalty split out from the
            # allocation benefit); that needs reward-shaping the prompt
            # doesn't specify, so it's left as a single shared signal here.
            #
            # FIX (stability): each raw reward is rescaled through this
            # server's per-decision running normalizer before being stored,
            # so allocation/migration/sharing rewards -- which have
            # different natural magnitudes -- end up on comparable, stable
            # scales for the critics.
            agent = self.agents[s.id]
            rewards[(s.id, cfg.ALLOCATION)] = agent.normalize_reward(cfg.ALLOCATION, allocation_reward)
            rewards[(s.id, cfg.MIGRATION)] = agent.normalize_reward(cfg.MIGRATION, migration_reward)
            rewards[(s.id, cfg.SHARING)] = agent.normalize_reward(cfg.SHARING, sharing_reward)
            # print("Hello")
            info, cons_time = s.get_info()
            
            # print(info, cons_time)
            infos.extend(list(info.values()))
            con_time.extend(cons_time)
            s.advance_slot()

        # 7) next-state snapshot. Recomputes local load post-execution
        # (finished/migrated tasks are gone) but reuses this slot's
        # combined neighbor load, since neighbors won't re-share until the
        # NEXT slot's local_load_sharing() call.
        for s in self.servers:
            s.local_load = s._compute_load_vector()
        next_state_tensors = {
            s.id: torch.as_tensor(self.agents[s.id].normalize_state(s.get_state()), dtype = torch.float32, device = cfg.DEVICE)
            for s in self.servers
        }

        # 8) store transitions
        for s in self.servers:
            for d in range(cfg.NUM_DECISIONS):
                key = (s.id, d)
                self.buffers[key].add(
                    state_tensors[s.id], raw_samples[key], log_probs[key],
                    entropies[key], rewards[key], next_state_tensors[s.id],
                )
        return infos, con_time