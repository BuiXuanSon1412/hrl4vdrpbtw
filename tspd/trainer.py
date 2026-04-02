import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import queue
import threading
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from actor_critic import TSPDActorCritic
from env import TSPDEnvironment, generate_tspd_instances
from optimizer import AdaBelief, CosineAnnealingWithCycles


@dataclass
class Experience:
    """Single training experience (trajectory)."""
    coords: torch.Tensor          # [B, N, 2]
    total_reward: torch.Tensor    # [B]
    log_probs: torch.Tensor       # [B]
    baseline: torch.Tensor        # [B]
    advantage: float              # Mean absolute advantage
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor


class PrioritizedExperienceBuffer:
    """
    Priority queue for high-advantage experiences.
    Experiences where |A| > τ are prioritized for immediate updates.
    Section 4.3: τ = 0.5
    """
    def __init__(self, maxsize: int = 64):
        self.actor_queue = queue.Queue(maxsize=maxsize)
        self.critic_queue = queue.Queue(maxsize=maxsize)
        self.priority_threshold = 0.5  # τ

    def put(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor, advantage: float):
        """Add losses to queues if priority threshold exceeded."""
        if abs(advantage) > self.priority_threshold:
            try:
                self.actor_queue.put_nowait(actor_loss.detach())
                self.critic_queue.put_nowait(critic_loss.detach())
            except queue.Full:
                pass  # Skip if queue is full

    def get_actor(self, timeout: float = 0.1) -> Optional[torch.Tensor]:
        try:
            return self.actor_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_critic(self, timeout: float = 0.1) -> Optional[torch.Tensor]:
        try:
            return self.critic_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ActorUpdateThread(threading.Thread):
    """
    Algorithm 2: ActorUpdateThread
    Asynchronously updates actor parameters from priority queue.
    """
    def __init__(
        self,
        model: TSPDActorCritic,
        optimizer: AdaBelief,
        scheduler: CosineAnnealingWithCycles,
        buffer: PrioritizedExperienceBuffer,
        device: torch.device,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.buffer = buffer
        self.device = device
        self.running = True
        self.grad_clip = 1.0

    def run(self):
        """Algorithm 2: while training not terminated, get loss and update."""
        while self.running:
            actor_loss = self.buffer.get_actor()
            if actor_loss is not None:
                # Recreate computation graph for backward (simplified - use stored loss)
                self.optimizer.zero_grad()
                # In practice, the full computation graph would be preserved
                # Here we use the pre-computed loss gradients
                if actor_loss.requires_grad:
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.encoder.parameters(), self.grad_clip
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.decoder.parameters(), self.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()

    def stop(self):
        self.running = False


class CriticUpdateThread(threading.Thread):
    """
    Algorithm 3: CriticUpdateThread
    Asynchronously updates critic parameters from priority queue.
    """

    def __init__(
        self,
        model: TSPDActorCritic,
        optimizer: AdaBelief,
        scheduler: CosineAnnealingWithCycles,
        buffer: PrioritizedExperienceBuffer,
        device: torch.device,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.buffer = buffer
        self.device = device
        self.running = True
        self.grad_clip = 1.0

    def run(self):
        """Algorithm 3: while training not terminated, get loss and update."""
        while self.running:
            critic_loss = self.buffer.get_critic()
            if critic_loss is not None:
                self.optimizer.zero_grad()
                if critic_loss.requires_grad:
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()

    def stop(self):
        self.running = False


class TSPDTrainer:
    """
    Main trainer implementing Algorithm 1 from the paper.
    
    Training Process:
    1. Generate training data
    2. Interact with environment (actor rollout)
    3. Compute advantage A = C(s,π) - V_φ(s)
    4. If |mean(A)| > τ: queue losses for async update
    5. Periodic validation
    6. Save best model
    """

    def __init__(
        self,
        n_nodes: int = 20,
        batch_size: int = 128,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_mgu_layers: int = 2,
        lr: float = 1e-4,
        n_epochs: int = 10000,
        n_cycles: int = 5,
        priority_threshold: float = 0.5,
        validate_every: int = 500,
        device: str = "cpu",
        instance_type: str = "random",
        drone_speed_ratio: float = 2.0,
    ):
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device)
        self.instance_type = instance_type
        self.validate_every = validate_every

        # Model
        self.model = TSPDActorCritic(
            n_nodes=n_nodes,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_mgu_layers=n_mgu_layers,
        ).to(self.device)

        # Environment
        self.env = TSPDEnvironment(
            n_nodes=n_nodes,
            batch_size=batch_size,
            drone_speed_ratio=drone_speed_ratio,
            device=device,
        )

        # Optimizers (AdaBelief, Section 4.3)
        self.actor_optimizer = AdaBelief(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0.01,
        )
        self.critic_optimizer = AdaBelief(
            self.model.critic.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0.01,
        )

        # Cosine annealing schedulers (5 cycles, Eq. 31)
        self.actor_scheduler = CosineAnnealingWithCycles(
            self.actor_optimizer, n_epochs, n_cycles
        )
        self.critic_scheduler = CosineAnnealingWithCycles(
            self.critic_optimizer, n_epochs, n_cycles
        )

        # Priority buffer
        self.buffer = PrioritizedExperienceBuffer()
        self.buffer.priority_threshold = priority_threshold

        # Training state
        self.best_reward = float('-inf')
        self.best_model_state = None
        self.training_history = {
            'rewards': [], 'epoch_times': [], 'cumulative_times': []
        }

        # Start async update threads
        self.actor_thread = ActorUpdateThread(
            self.model, self.actor_optimizer, self.actor_scheduler,
            self.buffer, self.device
        )
        self.critic_thread = CriticUpdateThread(
            self.model, self.critic_optimizer, self.critic_scheduler,
            self.buffer, self.device
        )

    def compute_losses(
        self,
        total_reward: torch.Tensor,
        log_probs: torch.Tensor,
        baseline: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute actor and critic losses.
        
        Actor loss (Eq. 25):
            L_actor = -1/B * Σ (C(s,π) - b(s)) * (log π_truck + log π_drone)
        
        Critic loss (Eq. 26):
            L_critic = 1/B * Σ ||C(s,π) - V_φ(s)||^2
        
        Advantage (Eq. 27):
            A(s, π) = C(s,π) - V_φ(s)
        """
        # Advantage A = actual_cost - baseline_estimate
        # Note: total_reward is negative cost, so actual_cost = -total_reward
        actual_cost = -total_reward  # [B] (positive values)

        # Advantage function (Eq. 27)
        advantage = actual_cost - baseline.detach()  # [B]

        # Actor loss (REINFORCE with baseline, Eq. 25)
        # L_actor = -mean(advantage * log_probs)
        actor_loss = (advantage * log_probs).mean()  # Note: log_probs already negative

        # Critic loss: MSE between estimate and actual cost (Eq. 26)
        critic_loss = F.mse_loss(baseline, actual_cost.detach())

        return actor_loss, critic_loss, advantage

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Single training epoch (Algorithm 1, lines 4-31).
        
        Returns:
            mean_reward: Average reward for this epoch
            epoch_time: Time taken
        """
        start_time = time.time()
        self.model.train()

        # Generate training data (Algorithm 1, line 6)
        coords = generate_tspd_instances(
            self.batch_size, self.n_nodes, self.instance_type,
            device=str(self.device)
        )

        # Reset environment (Algorithm 1, line 7)
        state = self.env.reset(coords)

        # Initialize MGU hidden state (Algorithm 1, line 8)
        # (handled internally by decoder)

        # Environment interaction + get trajectories (Algorithm 1, lines 13-17)
        total_reward, log_probs, baseline = self.model.forward(
            coords, self.env, state, greedy=False
        )

        # Compute losses (Algorithm 1, lines 18-20)
        actor_loss, critic_loss, advantage = self.compute_losses(
            total_reward, log_probs, baseline
        )

        # Advantage function A = R - V (Algorithm 1, line 18)
        mean_advantage = advantage.abs().mean().item()

        # Priority check: if |mean(A)| > τ, queue for update (Algorithm 1, lines 21-24)
        if mean_advantage > self.buffer.priority_threshold:
            # Direct update (simplified from full async threading)
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                max_norm=1.0
            )
            self.actor_optimizer.step()
            self.actor_scheduler.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), max_norm=1.0
            )
            self.critic_optimizer.step()
            self.critic_scheduler.step()

            # Also queue for async threads (for true A3C behavior)
            self.buffer.put(actor_loss.detach(), critic_loss.detach(), mean_advantage)

        epoch_time = time.time() - start_time
        mean_reward = total_reward.mean().item()

        return mean_reward, epoch_time

    def validate(self, n_val: int = 100, greedy: bool = True) -> float:
        """
        Validate model using greedy decoding.
        Returns mean reward (lower cost = better, so higher reward is better).
        """
        self.model.eval()
        total_rewards = []

        with torch.no_grad():
            for _ in range(max(1, n_val // self.batch_size)):
                coords = generate_tspd_instances(
                    self.batch_size, self.n_nodes, self.instance_type,
                    device=str(self.device)
                )
                state = self.env.reset(coords)
                total_reward, _, _ = self.model.forward(
                    coords, self.env, state, greedy=True
                )
                total_rewards.append(total_reward.mean().item())

        return np.mean(total_rewards)

    def train(self, verbose: bool = True) -> Dict:
        """
        Full training loop. Algorithm 1.
        
        Args:
            verbose: Print progress
        Returns:
            Training history
        """
        print(f"Starting A3C training for TSP-D (N={self.n_nodes})")
        print(f"Epochs: {self.n_epochs}, Batch: {self.batch_size}, Device: {self.device}")
        print("-" * 60)

        # Start async threads
        self.actor_thread.start()
        self.critic_thread.start()

        cumulative_time = 0.0

        try:
            for epoch in range(1, self.n_epochs + 1):
                # Train one epoch
                mean_reward, epoch_time = self.train_epoch(epoch)
                cumulative_time += epoch_time

                # Log every 200 epochs (as in paper)
                if epoch % 200 == 0 or epoch == 1:
                    self.training_history['rewards'].append(mean_reward)
                    self.training_history['epoch_times'].append(epoch_time)
                    self.training_history['cumulative_times'].append(cumulative_time)

                    if verbose:
                        avg_cost = -mean_reward
                        print(
                            f"Epoch {epoch:5d}/{self.n_epochs} | "
                            f"Cost: {avg_cost:.3f} | "
                            f"Time/epoch: {epoch_time:.3f}s | "
                            f"Total: {cumulative_time:.1f}s"
                        )

                # Periodic validation (Algorithm 1, lines 25-32)
                if epoch % self.validate_every == 0:
                    val_reward = self.validate()
                    val_cost = -val_reward

                    if val_reward > self.best_reward:
                        self.best_reward = val_reward
                        self.best_model_state = {
                            'epoch': epoch,
                            'model_state': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                            'actor_optimizer': self.actor_optimizer.state_dict(),
                            'critic_optimizer': self.critic_optimizer.state_dict(),
                            'val_cost': val_cost,
                        }
                        if verbose:
                            print(f"   New best model! Val cost: {val_cost:.4f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        finally:
            # Stop async threads
            self.actor_thread.stop()
            self.critic_thread.stop()

        print(f"\nTraining complete. Best val cost: {-self.best_reward:.4f}")
        return self.training_history

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save(self.best_model_state or self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Model loaded from {path}")

    def batch_sample_inference(
        self,
        coords: torch.Tensor,
        n_samples: int = 1200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch sampling inference strategy (as in experiments).
        Generate n_samples solutions and return the best.
        
        Section 5.1: Ours(sampling_k) strategy.
        
        Args:
            coords: [1, N, 2] or [B, N, 2] - single or batch instances
        Returns:
            best_costs: [B] - best solution costs
            best_actions: list of action sequences
        """
        self.model.eval()
        B = coords.shape[0]
        device = coords.device

        # Expand for sampling
        coords_expanded = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
        coords_expanded = coords_expanded.reshape(B * n_samples, self.n_nodes, 2)

        with torch.no_grad():
            state = self.env.reset(coords_expanded)
            total_reward, _, actions = self.model.forward(
                coords_expanded, self.env, state, greedy=False
            )

        # Reshape and find best
        total_reward = total_reward.view(B, n_samples)
        best_idx = total_reward.argmax(dim=-1)  # Higher reward = lower cost
        best_rewards = total_reward[torch.arange(B), best_idx]
        best_costs = -best_rewards

        return best_costs, best_idx
