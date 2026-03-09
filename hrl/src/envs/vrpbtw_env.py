"""
src/envs/vrpbtw_env.py
─────────────────────────────────────────────────────────────────────────────
Core Gymnasium environment for the Vehicle Routing Problem with
Backhauls and Time Windows (VRPBTW).

Responsibilities
────────────────
• State representation & observation space
• Action validation (feasibility checks)
• Transition dynamics (vehicle movement, load updates, time tracking)
• Reward delegation → RewardFunction (see src/rewards/)
• Episode termination / truncation logic

Design principles
─────────────────
• No neural-network code here — pure combinatorial problem logic.
• All problem-size parameters are injected via __init__; no globals.
• Instance data (customers, vehicles, drones) is re-sampled on every
  reset() call, making the env usable for both fixed and random instances.
• Supports variable-size instances via padding + mask in the observation
  so the same trained model can be applied to different sizes at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from pandas.io.common import tarfile

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────


class CustomerType(IntEnum):
    LINEHAUL = 0
    BACKHAUL = 1


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float  # positive = delivery; negative = pickup
    time_window_start: float
    time_window_end: float
    service_time: float
    customer_type: CustomerType


@dataclass
class Vehicle:
    id: int
    capacity: float
    current_load: float
    x: float
    y: float
    current_time: float
    route: List[int] = field(default_factory=list)
    visited_linehaul: bool = False
    visited_backhaul: bool = False


@dataclass
class Drone:
    id: int
    capacity: float
    battery_capacity: float
    current_battery: float
    speed: float
    is_available: bool
    x: float
    y: float


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


class VRPBTWEnv(gym.Env):
    """
    Gymnasium environment for VRPBTW with optional drone support.

    Observation
    ───────────
    A flat float32 vector containing (in order):
      • depot coordinates                     (2,)
      • customer features × num_customers     (num_customers × CUSTOMER_DIM,)
      • vehicle features × num_vehicles       (num_vehicles  × VEHICLE_DIM,)
      • drone features   × num_drones         (num_drones    × DRONE_DIM,)
    All values are normalized to [0, 1].

    Action
    ──────
    A dict:
      {
        "vehicle_id" : int,           # which vehicle to move
        "customer_id": int,           # target customer (-1 = return to depot)
        "use_drone"  : bool,
        "drone_id"   : int | None,
      }

    Termination
    ───────────
    • terminated = True  when all customers are served
    • truncated  = True  when step count exceeds max_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Feature dimensions (columns per entity)
    CUSTOMER_DIM = 7  # x, y, demand, tw_start, tw_end, type, served_flag
    VEHICLE_DIM = 7  # x, y, load_ratio, time, visited_lh, visited_bh, route_started
    DRONE_DIM = 5  # x, y, battery_ratio, available, capacity_ratio

    def __init__(
        self,
        num_customers: int = 100,
        num_vehicles: int = 7,
        num_drones: int = 7,
        vehicle_capacity: float = 100.0,
        drone_capacity: float = 10.0,
        drone_battery: float = 100.0,
        drone_speed: float = 2.0,
        map_size: float = 100.0,
        time_horizon: float = 200.0,
        linehaul_ratio: float = 0.5,
        max_steps_multiplier: int = 2,
        reward_fn=None,  # injectable RewardFunction (see src/rewards/)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # ── Problem parameters ──────────────────────────────────────────────
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.num_drones = num_drones
        self.vehicle_capacity = vehicle_capacity
        self.drone_capacity = drone_capacity
        self.drone_battery = drone_battery
        self.drone_speed = drone_speed
        self.map_size = map_size
        self.time_horizon = time_horizon
        self.linehaul_ratio = linehaul_ratio
        self.max_steps = num_customers * max_steps_multiplier
        self.render_mode = render_mode

        # ── Reward function (injected or default) ──────────────────────────
        # Decoupling reward from env allows easy swapping without subclassing.
        if reward_fn is None:
            from src.rewards.default import DefaultRewardFn

            reward_fn = DefaultRewardFn()
        self.reward_fn = reward_fn

        # ── Depot ──────────────────────────────────────────────────────────
        self.depot: Tuple[float, float] = (map_size / 2, map_size / 2)

        # ── Runtime state (populated by reset) ────────────────────────────
        self.customers: List[Customer] = []
        self.vehicles: List[Vehicle] = []
        self.drones: List[Drone] = []
        self.served_customers: set = set()
        self.step_count: int = 0
        self.total_cost: float = 0.0
        self.max_tardiness: float = 0.0

        # ── Gym spaces ────────────────────────────────────────────────────
        obs_dim = (
            2
            + num_customers * self.CUSTOMER_DIM
            + num_vehicles * self.VEHICLE_DIM
            + num_drones * self.DRONE_DIM
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # The env accepts dict actions; agents build these dicts themselves.
        self.action_space = spaces.Discrete(num_customers + 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Gym API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.served_customers = set()
        self.step_count = 0
        self.total_cost = 0.0
        self.max_tardiness = 0.0

        # Allow caller to inject a pre-built instance (e.g. from a benchmark file)
        if options and "instance" in options:
            self._load_instance(options["instance"])
        else:
            self._generate_instance(rng)

        return self._get_obs(), {}

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        vehicle_id = int(action["vehicle_id"])
        customer_id = int(action["customer_id"])
        use_drone = bool(action.get("use_drone", False))
        drone_id = action.get("drone_id", None)

        vehicle = self.vehicles[vehicle_id]

        # ── Depot return ───────────────────────────────────────────────────
        if customer_id == -1:
            reward, info = self._depot_return(vehicle)
        else:
            customer = self.customers[customer_id]
            error = self._validate(vehicle, customer, use_drone, drone_id)
            if error:
                reward = self.reward_fn.invalid_action(error)
                obs = self._get_obs()
                return obs, reward, False, False, {"error": error}

            if use_drone and drone_id is not None:
                reward, info = self._drone_delivery(vehicle, customer, int(drone_id))
            else:
                reward, info = self._vehicle_delivery(vehicle, customer)

        terminated = len(self.served_customers) == self.num_customers
        truncated = self.step_count >= self.max_steps

        # Apply unserved penalty on terminal step
        if (terminated or truncated) and len(
            self.served_customers
        ) < self.num_customers:
            n_unserved = self.num_customers - len(self.served_customers)
            reward += self.reward_fn.unserved_penalty(n_unserved)
            info["unserved_customers"] = n_unserved

        return self._get_obs(), reward, terminated, truncated, info

    def get_valid_actions(self, vehicle_id: int) -> List[int]:
        """
        Return list of feasible customer IDs for the given vehicle.
        Returns [-1] (depot only) when no customer is reachable.
        """
        vehicle = self.vehicles[vehicle_id]
        valid = []

        for c in self.customers:
            if c.id in self.served_customers:
                continue
            if c.customer_type == CustomerType.LINEHAUL:
                if vehicle.current_load >= c.demand:
                    valid.append(c.id)
            else:  # BACKHAUL
                if (
                    vehicle.visited_linehaul
                    and vehicle.current_load + c.demand <= vehicle.capacity
                ):
                    valid.append(c.id)

        # Depot return is valid only after the vehicle has started its route
        if len(vehicle.route) > 0:
            if not valid or len(self.served_customers) == self.num_customers:
                valid.append(-1)

        return valid if valid else [-1]

    # ─────────────────────────────────────────────────────────────────────────
    # Instance generation / loading
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_instance(self, rng: np.random.Generator):
        """Randomly generate a VRPBTW instance."""
        n_lh = int(self.num_customers * self.linehaul_ratio)

        self.customers = []
        for i in range(self.num_customers):
            ctype = CustomerType.LINEHAUL if i < n_lh else CustomerType.BACKHAUL
            tw_s = rng.uniform(0, self.time_horizon * 0.5)
            self.customers.append(
                Customer(
                    id=i,
                    x=rng.uniform(0, self.map_size),
                    y=rng.uniform(0, self.map_size),
                    demand=rng.uniform(5, 15),
                    time_window_start=tw_s,
                    time_window_end=tw_s + rng.uniform(30, 80),
                    service_time=rng.uniform(0.5, 2.0),
                    customer_type=ctype,
                )
            )

        self.vehicles = [
            Vehicle(
                id=i,
                capacity=self.vehicle_capacity,
                current_load=self.vehicle_capacity,
                x=self.depot[0],
                y=self.depot[1],
                current_time=0.0,
            )
            for i in range(self.num_vehicles)
        ]

        self.drones = [
            Drone(
                id=i,
                capacity=self.drone_capacity,
                battery_capacity=self.drone_battery,
                current_battery=self.drone_battery,
                speed=self.drone_speed,
                is_available=True,
                x=self.depot[0],
                y=self.depot[1],
            )
            for i in range(self.num_drones)
        ]

    def _load_instance(self, instance: Dict[str, Any]):
        """Load a pre-built instance dict (from benchmark files)."""
        self.customers = instance["customers"]
        self.vehicles = instance["vehicles"]
        self.drones = instance["drones"]

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        parts: List[float] = [
            self.depot[0] / self.map_size,
            self.depot[1] / self.map_size,
        ]
        for c in self.customers:
            parts.extend(
                [
                    c.x / self.map_size,
                    c.y / self.map_size,
                    c.demand / 20.0,
                    c.time_window_start / self.time_horizon,
                    c.time_window_end / self.time_horizon,
                    float(c.customer_type),
                    float(c.id in self.served_customers),
                ]
            )
        for v in self.vehicles:
            parts.extend(
                [
                    v.x / self.map_size,
                    v.y / self.map_size,
                    v.current_load / v.capacity,
                    v.current_time / self.time_horizon,
                    float(v.visited_linehaul),
                    float(v.visited_backhaul),
                    float(len(v.route) > 0),
                ]
            )
        for d in self.drones:
            parts.extend(
                [
                    d.x / self.map_size,
                    d.y / self.map_size,
                    d.current_battery / d.battery_capacity,
                    float(d.is_available),
                    d.capacity / self.drone_capacity,
                ]
            )
        return np.array(parts, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Transition helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(x1, y1, x2, y2) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    def _tardiness(self, arrival: float, c: Customer) -> float:
        if c.time_window_start <= arrival <= c.time_window_end:
            return 1.0
        deviation = min(
            abs(arrival - c.time_window_start),
            abs(arrival - c.time_window_end),
        )
        width = (c.time_window_end - c.time_window_start) / 2 + 1e-6
        return float(np.exp(-deviation / width))

    def _validate(
        self,
        vehicle: Vehicle,
        customer: Customer,
        use_drone: bool,
        drone_id: Optional[int],
    ) -> Optional[str]:
        if customer.id in self.served_customers:
            return "already_served"
        if customer.customer_type == CustomerType.LINEHAUL:
            if vehicle.current_load < customer.demand:
                return "insufficient_load"
        else:
            if not vehicle.visited_linehaul:
                return "must_visit_linehaul_first"
            if vehicle.current_load + customer.demand > vehicle.capacity:
                return "capacity_exceeded"
        if use_drone and drone_id is not None:
            drone = self.drones[drone_id]
            if not drone.is_available:
                return "drone_unavailable"
            if customer.demand > drone.capacity:
                return "drone_capacity_exceeded"
        return None

    def _depot_return(self, vehicle: Vehicle) -> Tuple[float, Dict]:
        dist = self._dist(vehicle.x, vehicle.y, *self.depot)
        vehicle.x, vehicle.y = self.depot
        vehicle.current_time += dist
        vehicle.current_load = self.vehicle_capacity
        self.total_cost += dist
        reward = self.reward_fn.travel(dist, 0.0)
        return reward, {"cost": dist, "tardiness": 0.0}

    def _vehicle_delivery(
        self, vehicle: Vehicle, customer: Customer
    ) -> Tuple[float, Dict]:
        dist = self._dist(vehicle.x, vehicle.y, customer.x, customer.y)
        vehicle.x, vehicle.y = customer.x, customer.y
        vehicle.current_time += dist + customer.service_time

        if customer.customer_type == CustomerType.LINEHAUL:
            vehicle.current_load -= customer.demand
            vehicle.visited_linehaul = True
        else:
            vehicle.current_load += customer.demand
            vehicle.visited_backhaul = True

        tardiness = self._tardiness(vehicle.current_time, customer)
        vehicle.route.append(customer.id)
        self.served_customers.add(customer.id)
        self.total_cost += dist
        self.max_tardiness = max(self.max_tardiness, tardiness)

        reward = self.reward_fn.travel(dist, tardiness)
        return reward, {"cost": dist, "tardiness": tardiness}

    def _drone_delivery(
        self, vehicle: Vehicle, customer: Customer, drone_id: int
    ) -> Tuple[float, Dict]:
        drone = self.drones[drone_id]
        launch_dist = self._dist(vehicle.x, vehicle.y, customer.x, customer.y)
        return_dist = self._dist(customer.x, customer.y, *self.depot)
        total_dist = launch_dist + return_dist
        battery_cost = total_dist * 0.5

        if battery_cost > drone.current_battery:
            reward = self.reward_fn.invalid_action("insufficient_battery")
            return reward, {"error": "insufficient_battery"}

        arrival = vehicle.current_time + launch_dist / drone.speed
        tardiness = self._tardiness(arrival, customer)
        cost = total_dist * 1.5  # drone travel is more expensive

        drone.current_battery -= battery_cost
        drone.x, drone.y = self.depot
        self.served_customers.add(customer.id)

        if customer.customer_type == CustomerType.LINEHAUL:
            vehicle.current_load -= customer.demand
            vehicle.visited_linehaul = True
        else:
            vehicle.current_load += customer.demand
            vehicle.visited_backhaul = True

        self.total_cost += cost
        self.max_tardiness = max(self.max_tardiness, tardiness)

        reward = self.reward_fn.travel(cost, tardiness)
        return reward, {"cost": cost, "tardiness": tardiness, "used_drone": True}

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics helpers (used by evaluation scripts)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def service_rate(self) -> float:
        return len(self.served_customers) / self.num_customers

    @property
    def avg_tardiness(self) -> float:
        n = len(self.served_customers)
        return self.max_tardiness / n if n > 0 else 0.0

    def render(self):
        if self.render_mode == "human":
            # Visualization delegated to src/utils/visualizer.py
            from .src.utils.visualizer import render_env

            render_env(self)
