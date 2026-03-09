"""
src/agents/hierarchical.py
─────────────────────────────────────────────────────────────────────────────
Hierarchical agent: high-level policy selects which vehicle to move;
low-level policy selects which customer to visit and whether to use a drone.

Neural network specifics live in src/models/ — this class only coordinates
the decision loop, experience collection, and training calls.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.models import build_model
from .base import BaseAgent


class HierarchicalAgent(BaseAgent):
    """
    Hierarchical Reinforcement Learning agent for VRPBTW.

    Components
    ──────────
    high_level_policy : HighLevelPolicy  — selects vehicle
    low_level_policy  : LowLevelPolicy   — selects customer + drone flag

    Both policies output logits; action selection is ε-greedy + temperature
    during training and greedy during evaluation.
    """

    def __init__(self, env, cfg):
        super().__init__(env, cfg)

        # ── Build neural networks ──────────────────────────────────────────
        self.high_level_policy, self.low_level_policy = build_model(cfg, env)

        # ── Optimisers ────────────────────────────────────────────────────
        self.hl_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(),
            lr=cfg.agent.high_level_lr,
        )
        self.ll_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(),
            lr=cfg.agent.low_level_lr,
        )

        # ── Exploration ───────────────────────────────────────────────────
        self.epsilon = cfg.agent.epsilon_start
        self.temperature = cfg.agent.temperature_start
        self._eps_start = cfg.agent.epsilon_start
        self._eps_end = cfg.agent.epsilon_end
        self._eps_decay = cfg.agent.epsilon_decay
        self._tmp_start = cfg.agent.temperature_start
        self._tmp_end = cfg.agent.temperature_end
        self._tmp_decay = cfg.agent.temperature_decay

        # ── Experience buffers (simple lists; swap for PER if desired) ────
        self.hl_buffer: List[Dict] = []
        self.ll_buffer: List[Dict] = []

        # ── Running stats ─────────────────────────────────────────────────
        self.train_stats: Dict[str, List[float]] = {
            "hl_loss": [],
            "ll_loss": [],
            "entropy": [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BaseAgent interface
    # ─────────────────────────────────────────────────────────────────────────

    def rollout(self, training: bool = True) -> Dict[str, Any]:
        """Run one complete episode, collecting experience if training=True."""
        self.high_level_policy.eval()
        self.low_level_policy.eval()

        obs, _ = self.env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            vehicle_id = self._select_vehicle(obs, training)
            valid_actions = self.env.get_valid_actions(vehicle_id)

            if valid_actions == [-1] or valid_actions[0] == -1:
                customer_id, use_drone = -1, False
            else:
                customer_id, use_drone = self._select_customer(
                    obs, vehicle_id, valid_actions, training
                )

            action = {
                "vehicle_id": vehicle_id,
                "customer_id": customer_id,
                "use_drone": use_drone,
                "drone_id": 0 if use_drone else None,
            }

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if training:
                self._store_experience(
                    obs,
                    vehicle_id,
                    customer_id,
                    use_drone,
                    reward,
                    next_obs,
                    terminated or truncated,
                )

            total_reward += reward
            obs = next_obs
            steps += 1

        return {
            "total_reward": total_reward,
            "total_cost": self.env.total_cost,
            "max_tardiness": self.env.max_tardiness,
            "service_rate": self.env.service_rate,
            "customers_served": len(self.env.served_customers),
            "steps": steps,
        }

    def train_step(self) -> Dict[str, float]:
        """Sample from buffers and do one gradient update."""
        batch_size = self.cfg.training.batch_size
        if len(self.hl_buffer) < batch_size or len(self.ll_buffer) < batch_size:
            return {}

        self.high_level_policy.train()
        self.low_level_policy.train()

        hl_loss, hl_ent = self._train_high_level(batch_size)
        ll_loss, ll_ent = self._train_low_level(batch_size)

        self.train_stats["hl_loss"].append(hl_loss)
        self.train_stats["ll_loss"].append(ll_loss)
        self.train_stats["entropy"].append(ll_ent)

        return {
            "hl_loss": hl_loss,
            "hl_entropy": hl_ent,
            "ll_loss": ll_loss,
            "ll_entropy": ll_ent,
        }

    def update_exploration(self, episode: int) -> None:
        self.epsilon = max(
            self._eps_end,
            self._eps_start * (self._eps_decay**episode),
        )
        self.temperature = max(
            self._tmp_end,
            self._tmp_start * (self._tmp_decay**episode),
        )

    def clear_buffers(self) -> None:
        self.hl_buffer.clear()
        self.ll_buffer.clear()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hl_state": self.high_level_policy.state_dict(),
                "ll_state": self.low_level_policy.state_dict(),
                "hl_optim": self.hl_optimizer.state_dict(),
                "ll_optim": self.ll_optimizer.state_dict(),
                "epsilon": self.epsilon,
                "temperature": self.temperature,
                "train_stats": self.train_stats,
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location="cpu")
        self.high_level_policy.load_state_dict(ckpt["hl_state"])
        self.low_level_policy.load_state_dict(ckpt["ll_state"])
        self.hl_optimizer.load_state_dict(ckpt["hl_optim"])
        self.ll_optimizer.load_state_dict(ckpt["ll_optim"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.temperature = ckpt.get("temperature", self.temperature)
        self.train_stats = ckpt.get("train_stats", self.train_stats)

    @property
    def num_parameters(self) -> int:
        total = sum(p.numel() for p in self.high_level_policy.parameters())
        total += sum(p.numel() for p in self.low_level_policy.parameters())
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers (action selection)
    # ─────────────────────────────────────────────────────────────────────────

    def _select_vehicle(self, obs: np.ndarray, training: bool) -> int:
        """High-level: pick which vehicle to route next."""
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = self.high_level_policy(state_t)

        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.env.num_vehicles))

        if training:
            dist = torch.distributions.Categorical(logits=logits / self.temperature)
            return int(dist.sample().item())

        return int(logits.argmax(dim=-1).item())

    def _select_customer(
        self,
        obs: np.ndarray,
        vehicle_id: int,
        valid_actions: List[int],
        training: bool,
    ):
        """Low-level: pick customer and drone flag."""
        # Feature extraction delegated to model
        with torch.no_grad():
            cust_feat = self._extract_customer_features(obs)
            veh_ctx = self._extract_vehicle_context(obs, vehicle_id)
            mask = self._build_mask(valid_actions)

            c_logits, d_logits, _ = self.low_level_policy(cust_feat, veh_ctx, mask)

        if training and np.random.random() < self.epsilon:
            cid = int(np.random.choice(valid_actions))
            use_drone = bool(np.random.randint(2))
            return cid, use_drone

        # Customer selection
        valid_c_logits = c_logits[0][valid_actions]
        if training:
            probs = torch.softmax(valid_c_logits / self.temperature, dim=0)
            idx = int(torch.distributions.Categorical(probs).sample().item())
        else:
            idx = int(valid_c_logits.argmax().item())
        cid = valid_actions[idx]

        # Drone selection
        use_drone = bool(d_logits.argmax(dim=-1).item())
        return cid, use_drone

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction (obs → tensors for the policies)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_customer_features(self, obs: np.ndarray) -> torch.Tensor:
        start = 2
        end = 2 + self.env.num_customers * self.env.CUSTOMER_DIM
        feat = obs[start:end].reshape(self.env.num_customers, self.env.CUSTOMER_DIM)
        return torch.FloatTensor(feat).unsqueeze(0)

    def _extract_vehicle_context(
        self, obs: np.ndarray, vehicle_id: int
    ) -> torch.Tensor:
        base = 2 + self.env.num_customers * self.env.CUSTOMER_DIM
        start = base + vehicle_id * self.env.VEHICLE_DIM
        end = start + self.env.VEHICLE_DIM
        return torch.FloatTensor(obs[start:end]).unsqueeze(0)

    def _build_mask(self, valid_actions: List[int]) -> torch.Tensor:
        mask = torch.ones(1, self.env.num_customers, dtype=torch.bool)
        for a in valid_actions:
            if a >= 0:
                mask[0, a] = False
        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Training (REINFORCE + value baseline)
    # ─────────────────────────────────────────────────────────────────────────

    def _train_high_level(self, batch_size: int):
        import torch.nn.functional as F

        idxs = np.random.choice(len(self.hl_buffer), batch_size, replace=False)
        batch = [self.hl_buffer[i] for i in idxs]

        states = torch.FloatTensor([b["state"] for b in batch])
        veh_ids = torch.LongTensor([b["vehicle_id"] for b in batch])
        rewards = torch.FloatTensor([b["reward"] for b in batch])

        logits, values = self.high_level_policy(states)
        advantages = rewards - values.squeeze().detach()

        log_probs = F.log_softmax(logits, dim=-1)
        sel_lp = log_probs.gather(1, veh_ids.unsqueeze(1)).squeeze()
        pol_loss = -(sel_lp * advantages).mean()

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()
        val_loss = F.mse_loss(values.squeeze(), rewards)

        loss = pol_loss - self.cfg.agent.entropy_coef * entropy + 0.5 * val_loss

        self.hl_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.high_level_policy.parameters(), self.cfg.training.grad_clip
        )
        self.hl_optimizer.step()

        return loss.item(), entropy.item()

    def _train_low_level(self, batch_size: int):
        import torch.nn.functional as F

        valid = [b for b in self.ll_buffer if b["customer_features"] is not None]
        if len(valid) < batch_size:
            return 0.0, 0.0

        idxs = np.random.choice(len(valid), batch_size, replace=False)
        batch = [valid[i] for i in idxs]

        c_feat = torch.FloatTensor([b["customer_features"] for b in batch])
        v_ctx = torch.FloatTensor([b["vehicle_context"] for b in batch])
        c_ids = torch.LongTensor([b["customer_id"] for b in batch])
        drones = torch.LongTensor([int(b["use_drone"]) for b in batch])
        rewards = torch.FloatTensor([b["reward"] for b in batch])
        mask = torch.zeros(batch_size, self.env.num_customers, dtype=torch.bool)

        c_logits, d_logits, values = self.low_level_policy(c_feat, v_ctx, mask)
        advantages = rewards - values.squeeze().detach()

        c_lp = F.log_softmax(c_logits, dim=-1)
        sel_c = c_lp.gather(1, c_ids.unsqueeze(1)).squeeze()
        c_loss = -(sel_c * advantages).mean()
        c_ent = -(F.softmax(c_logits, dim=-1) * c_lp).sum(-1).mean()

        d_lp = F.log_softmax(d_logits, dim=-1)
        sel_d = d_lp.gather(1, drones.unsqueeze(1)).squeeze()
        d_loss = -(sel_d * advantages).mean()
        d_ent = -(F.softmax(d_logits, dim=-1) * d_lp).sum(-1).mean()

        entropy = c_ent + d_ent
        val_loss = F.mse_loss(values.squeeze(), rewards)
        loss = (
            c_loss
            + 0.5 * d_loss
            - self.cfg.agent.entropy_coef * entropy
            + 0.5 * val_loss
        )

        self.ll_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.low_level_policy.parameters(), self.cfg.training.grad_clip
        )
        self.ll_optimizer.step()

        return loss.item(), entropy.item()

    def _store_experience(
        self, obs, vehicle_id, customer_id, use_drone, reward, next_obs, done
    ):
        self.hl_buffer.append(
            {
                "state": obs,
                "vehicle_id": vehicle_id,
                "reward": reward,
                "next_state": next_obs,
                "done": done,
            }
        )
        c_feat = (
            self._extract_customer_features(obs).squeeze(0).numpy()
            if customer_id != -1
            else None
        )
        v_ctx = (
            self._extract_vehicle_context(obs, vehicle_id).squeeze(0).numpy()
            if customer_id != -1
            else None
        )
        self.ll_buffer.append(
            {
                "customer_features": c_feat,
                "vehicle_context": v_ctx,
                "customer_id": customer_id,
                "use_drone": use_drone,
                "reward": reward,
                "done": done,
            }
        )
