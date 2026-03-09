"""
tests/test_env.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for VRPBTWEnv.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.envs.vrpbtw_env import VRPBTWEnv, CustomerType
from src.rewards.default import DefaultRewardFn


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def small_env():
    reward_fn = DefaultRewardFn()
    env = VRPBTWEnv(
        num_customers=10,
        num_vehicles=2,
        num_drones=1,
        vehicle_capacity=100.0,
        map_size=100.0,
        reward_fn=reward_fn,
    )
    env.reset(seed=42)
    return env


# ── Observation space ─────────────────────────────────────────────────────────


def test_obs_shape(small_env):
    obs, _ = small_env.reset()
    assert obs.shape == small_env.observation_space.shape


def test_obs_range(small_env):
    obs, _ = small_env.reset()
    assert obs.min() >= 0.0, "Observation contains values below 0"
    assert obs.max() <= 1.0 + 1e-5, "Observation contains values above 1"


def test_obs_dtype(small_env):
    obs, _ = small_env.reset()
    assert obs.dtype == np.float32


# ── Episode dynamics ──────────────────────────────────────────────────────────


def test_reset_clears_state(small_env):
    obs1, _ = small_env.reset(seed=1)
    obs2, _ = small_env.reset(seed=1)
    np.testing.assert_array_equal(obs1, obs2)


def test_reset_different_seeds(small_env):
    obs1, _ = small_env.reset(seed=1)
    obs2, _ = small_env.reset(seed=2)
    assert not np.array_equal(obs1, obs2)


def test_step_returns_correct_keys(small_env):
    valid = small_env.get_valid_actions(0)
    cid = valid[0] if valid[0] != -1 else valid[-1]
    action = {"vehicle_id": 0, "customer_id": cid, "use_drone": False, "drone_id": None}
    result = small_env.step(action)
    obs, reward, terminated, truncated, info = result
    assert obs.shape == small_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_depot_return_action(small_env):
    # Force a route, then return to depot
    env = small_env
    obs, _ = env.reset(seed=0)

    # First serve a linehaul customer
    valid = env.get_valid_actions(0)
    lh_actions = [
        a
        for a in valid
        if a >= 0 and env.customers[a].customer_type == CustomerType.LINEHAUL
    ]
    if not lh_actions:
        pytest.skip("No valid linehaul action in this random seed")

    action = {
        "vehicle_id": 0,
        "customer_id": lh_actions[0],
        "use_drone": False,
        "drone_id": None,
    }
    env.step(action)

    # Now depot return should be available
    valid_after = env.get_valid_actions(0)
    if len(valid_after) == 1 and valid_after[0] == -1:
        action2 = {
            "vehicle_id": 0,
            "customer_id": -1,
            "use_drone": False,
            "drone_id": None,
        }
        obs2, reward2, *_ = env.step(action2)
        assert isinstance(reward2, float)


# ── Validity checks ───────────────────────────────────────────────────────────


def test_invalid_action_already_served(small_env):
    """Serving the same customer twice should return an error in info."""
    env = small_env
    valid = env.get_valid_actions(0)
    lh = [
        a
        for a in valid
        if a >= 0 and env.customers[a].customer_type == CustomerType.LINEHAUL
    ]
    if not lh:
        pytest.skip("No valid linehaul action")

    cid = lh[0]
    action = {"vehicle_id": 0, "customer_id": cid, "use_drone": False, "drone_id": None}
    env.step(action)

    # Try serving the same customer again
    obs, reward, *_, info = env.step(action)
    assert "error" in info


def test_backhaul_before_linehaul_invalid(small_env):
    """Backhaul visit before any linehaul visit must be rejected."""
    env = small_env
    bh_customers = [
        c for c in env.customers if c.customer_type == CustomerType.BACKHAUL
    ]
    if not bh_customers:
        pytest.skip("No backhaul customers in this seed")

    action = {
        "vehicle_id": 0,
        "customer_id": bh_customers[0].id,
        "use_drone": False,
        "drone_id": None,
    }
    obs, reward, *_, info = env.step(action)
    assert "error" in info


# ── Service rate tracking ─────────────────────────────────────────────────────


def test_service_rate_increases(small_env):
    env = small_env
    env.reset(seed=7)
    assert env.service_rate == 0.0

    valid = env.get_valid_actions(0)
    lh = [
        a
        for a in valid
        if a >= 0 and env.customers[a].customer_type == CustomerType.LINEHAUL
    ]
    if not lh:
        pytest.skip("No valid linehaul action")

    action = {
        "vehicle_id": 0,
        "customer_id": lh[0],
        "use_drone": False,
        "drone_id": None,
    }
    env.step(action)
    assert env.service_rate > 0.0


# ── Reward function ───────────────────────────────────────────────────────────


def test_reward_is_finite(small_env):
    env = small_env
    valid = env.get_valid_actions(0)
    lh = [a for a in valid if a >= 0]
    if not lh:
        pytest.skip("No valid action")

    action = {
        "vehicle_id": 0,
        "customer_id": lh[0],
        "use_drone": False,
        "drone_id": None,
    }
    _, reward, *_ = env.step(action)
    assert np.isfinite(reward)


def test_sparse_reward_fn():
    from src.rewards.default import SparseRewardFn

    fn = SparseRewardFn()
    assert fn.travel(100, 1.0) == 0.0
    assert fn.invalid_action("x") < 0
    assert fn.unserved_penalty(3) < 0
