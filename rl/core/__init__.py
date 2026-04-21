# core/__init__.py
# Convenience re-exports so callers can write:
#   from core import BaseAgent, PPOAgent, RolloutBuffer, Environment, ...
# instead of drilling into sub-modules.

from core.agent import Agent, BaseAgent
from core.buffer import Transition, RolloutBuffer
from core.estimator import BaseEstimator, PPOEstimator
from core.evaluator import Evaluator
from core.logger import Logger
from core.policy import BasePolicy
from core.environment import Environment, ActionMask, StepResult, Solution, SolutionPool
from core.task import Task, SimpleTask, TaskManager
from core.trainer import (
    Trainer,
    MetaTrainer,
    POMOTrainer,
    CurriculumScheduler,
    FineTuner,
)
from core.utils import SeedManager, RunningNormalizer

__all__ = [
    # agents
    "Agent",
    "BaseAgent",
    # buffers
    "Transition",
    "RolloutBuffer",
    # curriculum
    "CurriculumScheduler",
    # estimators
    "BaseEstimator",
    "PPOEstimator",
    # evaluator
    "Evaluator",
    # fine tuner
    "FineTuner",
    # logger
    "Logger",
    # networks
    "BasePolicy",
    # problem
    "Environment",
    "ActionMask",
    "StepResult",
    "Solution",
    "SolutionPool",
    # task management
    "Task",
    "SimpleTask",
    "TaskManager",
    # trainer
    "Trainer",
    "MetaTrainer",
    "POMOTrainer",
    # utils
    "SeedManager",
    "RunningNormalizer",
]
