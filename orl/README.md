# RL Solver for Combinatorial Optimisation Problems

A reusable, end-to-end Reinforcement Learning framework that covers
**every stage of the RL pipeline** for arbitrary combinatorial problems.

---

## Architecture

```
rl_combinatorial/
│
├── core/
│   ├── problem.py          ← Abstract CombinatorialProblem (MDP definition)
│   ├── solution.py         ← Solution + SolutionPool containers
│   └── buffers.py          ← ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
│
├── environments/
│   └── combinatorial_env.py← Gym-style MDP wrapper with reward shaping
│
├── networks/
│   └── policy_network.py   ← Encoder–Decoder attention network (+ NumPy fallback)
│
├── agents/
│   ├── ppo_agent.py        ← On-policy PPO (rollout → GAE → clipped update)
│   └── dqn_agent.py        ← Off-policy Double DQN (+ optional PER)
│
├── training/
│   ├── trainer.py          ← Master training loop (curriculum, logging, checkpointing)
│   └── evaluator.py        ← Greedy / sampling / beam-search decoding + metrics
│
├── utils/
│   └── helpers.py          ← MetricsTracker, EpsilonScheduler, LRScheduler, …
│
└── examples/
    ├── knapsack_problem.py ← Concrete CombinatorialProblem implementation (0/1 Knapsack)
    └── train_knapsack.py   ← Full training script demonstrating all stages
```

---

## Full RL Procedure Covered

| Stage | Location | Description |
|---|---|---|
| **1. Problem Formulation** | `core/problem.py` | State, action, reward, termination, observation |
| **2. MDP Environment** | `environments/combinatorial_env.py` | Episode lifecycle, action masking, reward shaping |
| **3. Policy Network** | `networks/policy_network.py` | Transformer encoder + decoder attention head + critic |
| **4. Experience Collection** | `agents/ppo_agent.py`, `dqn_agent.py` | Rollout buffer (PPO) / replay buffer (DQN) |
| **5. Advantage Estimation** | `core/buffers.py` | GAE-λ computation in RolloutBuffer |
| **6. Policy Update** | `agents/ppo_agent.py` | PPO-clip + value loss + entropy bonus |
| **7. Q-Learning Update** | `agents/dqn_agent.py` | Double DQN + PER + soft/hard target sync |
| **8. Exploration** | `agents/dqn_agent.py`, `utils/helpers.py` | ε-greedy with linear decay |
| **9. Training Loop** | `training/trainer.py` | Curriculum scheduling, logging, early stopping |
| **10. Evaluation** | `training/evaluator.py` | Greedy / sampling / beam-search decoding |
| **11. Checkpointing** | `agents/*.py`, `training/trainer.py` | Save/load best model, periodic snapshots |

---

## Quickstart

### 1. Define your problem (one class, ~7 methods)

```python
from core.problem import CombinatorialProblem, ActionMask, StepResult

class MyProblem(CombinatorialProblem):
    def encode_instance(self, raw):   ...  # parse input
    def initial_state(self):          ...  # empty solution
    def get_action_mask(self, state): ...  # feasibility
    def apply_action(self, state, a): ...  # transition + reward
    def state_to_obs(self, state):    ...  # numpy observation
    def evaluate(self, state):        ...  # objective value
    def is_complete(self, state):     ...  # termination check
    action_space_size = ...                # int property
    observation_shape = ...                # tuple property
```

### 2. Wrap in an environment

```python
from environments.combinatorial_env import CombinatorialEnv

env = CombinatorialEnv(problem=MyProblem(), max_steps=100,
                       subtract_baseline=True, dense_shaping=True)
```

### 3. Create an agent

```python
from agents.ppo_agent import PPOAgent, PPOConfig

agent = PPOAgent(
    obs_shape=env.problem.observation_shape,
    action_space_size=env.problem.action_space_size,
    cfg=PPOConfig(lr=3e-4, rollout_len=1024),
)
```

### 4. Train

```python
from training.trainer import Trainer, TrainerConfig

trainer = Trainer(agent, env, instance_generator=my_gen_fn,
                  cfg=TrainerConfig(total_timesteps=500_000))
trainer.train()
```

### 5. Evaluate with beam search

```python
from training.evaluator import Evaluator

evaluator = Evaluator(agent, env, beam_width=5, n_episodes=50)
stats = evaluator.evaluate(my_gen_fn)
print(stats)
```

---

## Running the Knapsack Example

```bash
# PPO training (no GPU needed)
python examples/train_knapsack.py --steps 30000

# DQN training
python examples/train_knapsack.py --dqn --steps 30000

# Beam search decoding (width=3)
python examples/train_knapsack.py --beam 3
```

---

## Key Design Decisions

- **Action masking** is enforced at every level (env, agent, network) ensuring
  the policy never selects infeasible actions.
- **Framework-agnostic core**: `core/` and `environments/` are pure NumPy;
  PyTorch is only required for gradient-based training.
- **Reward shaping**: baseline subtraction, dense vs. sparse, and reward scaling
  are all configurable in `CombinatorialEnv`.
- **Pluggable agents**: PPO and DQN share the same `select_action / save / load`
  interface, making it trivial to swap algorithms.
- **Curriculum learning**: problem size grows automatically as training progresses.
