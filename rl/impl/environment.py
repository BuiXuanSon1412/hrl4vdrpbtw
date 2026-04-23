"""
problems/vrpbtw.py
------------------
VRPBTW with heterogeneous truck-drone fleets.

Reward design
-------------
Per-step reward uses potential-based shaping:
    r_t = -(f(s_{t+1}) - f(s_t))

where f(s) = total_cost + c_t * max_dist * N * (N - k)
    total_cost: total travel cost
    N: total customers
    k: served customers

This objective prioritizes serving customers (lexicographic) while minimizing cost.
Penalties are maintained incrementally in VRPBTWState.

At termination:
    feasible   -> r_T = -(f(s_T) - f(s_{T-1}))  (potential-based shaping)
    infeasible -> r_T = -1e6                     (hard penalty for infeasible end)

Constants
---------
NODE_FEAT_DIM  = 5   [x, y, demand, tw_open, tw_close]
VEH_FEAT_DIM   = 5   [x, y, load, time, deadline]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.environment import (
    ActionMask,
    Environment,
    Solution,
    StepResult,
)

# ---------------------------------------------------------------------------
# Feature-dimension constants
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 5
VEH_FEAT_DIM = 5

TRUCK = 0
DRONE = 1
DEPOT = 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # Per fleet (K,)
    truck_node: np.ndarray
    truck_time: np.ndarray
    truck_load: np.ndarray
    truck_phase: np.ndarray

    drone_node: np.ndarray
    drone_time: np.ndarray
    drone_load: np.ndarray
    drone_launch_time: np.ndarray
    drone_active: np.ndarray
    drone_launch_node: np.ndarray  # (K,) truck node at which drone k last launched

    # Global
    served: np.ndarray  # (N+1,) bool

    # Incremental objective accumulators
    current_cost: float  # total travel cost so far
    current_max_tard: float  # max tardiness across served customers so far

    # Routes (for logging + evaluate())
    truck_routes: List[List[int]]
    drone_route_nodes: List[List[int]]
    drone_route_mask: List[List[int]]


def _copy_state(s: VRPBTWState) -> VRPBTWState:
    return VRPBTWState(
        truck_node=s.truck_node.copy(),
        truck_time=s.truck_time.copy(),
        truck_load=s.truck_load.copy(),
        truck_phase=s.truck_phase.copy(),
        drone_node=s.drone_node.copy(),
        drone_time=s.drone_time.copy(),
        drone_load=s.drone_load.copy(),
        drone_launch_time=s.drone_launch_time.copy(),
        drone_active=s.drone_active.copy(),
        drone_launch_node=s.drone_launch_node.copy(),
        served=s.served.copy(),
        current_cost=s.current_cost,
        current_max_tard=s.current_max_tard,
        truck_routes=[list(r) for r in s.truck_routes],
        drone_route_nodes=[list(r) for r in s.drone_route_nodes],
        drone_route_mask=[list(m) for m in s.drone_route_mask],
    )


# ---------------------------------------------------------------------------
# VRPBTWProblem
# ---------------------------------------------------------------------------


class VRPBTWEnv(Environment):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(name="VRPBTW")
        # Instance parameters (set dynamically in reset via encode_instance)
        self.n_customers: int = 10
        self.n_fleets: int = 2
        self.K: int = 2

        # Static node feature arrays (placeholder, resized in encode_instance)
        self.coords: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.demands: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_open: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_close: np.ndarray = np.zeros(1, dtype=np.float32)
        self.service_times: np.ndarray = np.zeros(1, dtype=np.float32)
        self.manhattan_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.euclidean_dist: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        # Static environment parameters from config
        self.max_coord: float = float(cfg.get("max_coord", 100.0))
        self.Q_t: float = float(cfg.get("capacity_truck", 200.0))
        self.Q_d: float = float(cfg.get("capacity_drone", 20.0))
        self.T_max: float = float(cfg.get("t_max_system_h", 24.0))
        self.t_max: float = float(cfg.get("drone_duration_h", 1.0))
        self.v_t: float = float(cfg.get("v_truck_km_h", 40.0))
        self.v_d: float = float(cfg.get("v_drone_km_h", 60.0))
        self.c_t: float = float(cfg.get("truck_cost_unit", 1.0))
        self.c_d: float = float(cfg.get("drone_cost_unit", 0.5))
        self.launch_time: float = float(cfg.get("drone_takeoff_min", 1.0)) / 60.0
        self.land_time: float = float(cfg.get("drone_landing_min", 1.0)) / 60.0

        self._linehaul_idx: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_idx: np.ndarray = np.array([], dtype=np.int32)

    @classmethod
    def from_config(cls, cfg: Dict) -> "VRPBTWEnv":
        """Factory method: instantiate VRPBTWEnv from config dict.

        Instance-specific parameters (n_customers, n_fleets) are set dynamically
        in reset() via encode_instance() when raw_instance is provided.
        """
        # Support both old (flat) and new (properties) structure
        props = cfg.get("properties", cfg)
        return cls(props)

    # ------------------------------------------------------------------
    # encode_instance
    # ------------------------------------------------------------------

    def encode_instance(self, raw_instance: Dict) -> None:
        depot = np.array(raw_instance["depot"], dtype=np.float32)
        customers = np.array(raw_instance["customers"], dtype=np.float32)

        self.n_customers = len(customers)
        self.K = int(raw_instance["n_fleets"])
        self.n_fleets = self.K

        coords_all = np.vstack([depot, customers[:, :2]])
        tw_open_all = np.concatenate([[0.0], customers[:, 2]])
        tw_close_all = np.concatenate(
            [[float(raw_instance["system_duration"])], customers[:, 3]]
        )
        demands_all = np.concatenate([[0.0], customers[:, 4]])
        svc = float(raw_instance.get("service_time", 0.0))
        svc_all = np.full(self.n_customers + 1, svc, dtype=np.float32)
        svc_all[DEPOT] = 0.0

        self.coords = coords_all.astype(np.float32)
        self.tw_open = tw_open_all.astype(np.float32)
        self.tw_close = tw_close_all.astype(np.float32)
        self.demands = demands_all.astype(np.float32)
        self.service_times = svc_all

        dx = self.coords[:, None, 0] - self.coords[None, :, 0]
        dy = self.coords[:, None, 1] - self.coords[None, :, 1]
        self.euclidean_dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
        self.manhattan_dist = (np.abs(dx) + np.abs(dy)).astype(np.float32)

        self.Q_t = float(raw_instance["truck_capacity"])
        self.Q_d = float(raw_instance["drone_capacity"])
        self.T_max = float(raw_instance["system_duration"])
        self.t_max = float(raw_instance["trip_duration"])
        self.v_t = float(raw_instance["truck_speed"])
        self.v_d = float(raw_instance["drone_speed"])
        self.c_t = float(raw_instance["truck_cost"])
        self.c_d = float(raw_instance["drone_cost"])
        self.launch_time = float(raw_instance.get("launch_time", 0.0))
        self.land_time = float(raw_instance.get("land_time", 0.0))
        self.max_coord = float(raw_instance.get("max_coord", 100.0))

        cust_d = self.demands[1:]
        self._linehaul_idx = (np.where(cust_d > 0)[0] + 1).astype(np.int32)
        self._backhaul_idx = (np.where(cust_d < 0)[0] + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # reset override — computes nadir internally before returning state
    # ------------------------------------------------------------------

    def reset(self, raw_instance: Any) -> Tuple[Dict, Dict[str, Any]]:
        """
        Encode the instance and return the first observation.

        Returns (obs, info) tuple where obs is the state dict and info
        contains action_mask and feasible_actions.
        """
        self.encode_instance(raw_instance)
        self._n_steps = 0
        self._current_state = self.initial_state()
        mask = self.get_action_mask(self._current_state)
        obs = self.state_to_obs(self._current_state)
        return obs, {
            "action_mask": mask.mask,
            "feasible_actions": mask.action_indices,
        }

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def _compute_objective(self, cost: float, served_count: int) -> float:
        """
        New lexicographic objective: minimize cost, prioritize serving customers.
        f = total_cost + c_t * max_dist * N * (N - k)

        Where:
        - total_cost: total travel cost
        - c_t: truck cost unit
        - max_dist: maximum distance (2 * max_coord)
        - N: total customers
        - k: served customers
        - (N-k): unserved customers

        This ensures k+1 served customers always beats k served customers,
        even with worst-case additional distance.
        """
        N = self.n_customers
        k = served_count
        max_dist = 2.0 * self.max_coord
        unserved_penalty = self.c_t * max_dist * N * (N - k)
        return cost + unserved_penalty


    # ------------------------------------------------------------------
    # initial_state
    # ------------------------------------------------------------------

    def initial_state(self) -> VRPBTWState:
        K = self.K
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        return VRPBTWState(
            truck_node=np.zeros(K, dtype=np.int32),
            truck_time=np.zeros(K, dtype=np.float32),
            truck_load=np.full(K, self.Q_t, dtype=np.float32),
            truck_phase=np.zeros(K, dtype=np.int32),
            drone_node=np.zeros(K, dtype=np.int32),
            drone_time=np.zeros(K, dtype=np.float32),
            drone_load=np.full(K, self.Q_d, dtype=np.float32),
            drone_launch_time=np.zeros(K, dtype=np.float32),
            drone_active=np.zeros(K, dtype=bool),
            drone_launch_node=np.zeros(K, dtype=np.int32),
            served=served,
            current_cost=0.0,
            current_max_tard=0.0,
            truck_routes=[[] for _ in range(K)],
            drone_route_nodes=[[] for _ in range(K)],
            drone_route_mask=[[] for _ in range(K)],
        )

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def encode_action(self, node: int, vehicle_idx: int) -> int:
        return node * (2 * self.K) + vehicle_idx

    def decode_action(self, action: int) -> Tuple[int, int]:
        return action // (2 * self.K), action % (2 * self.K)

    def vehicle_fleet_type(self, vehicle_idx: int) -> Tuple[int, int]:
        if vehicle_idx < self.K:
            return vehicle_idx, TRUCK
        return vehicle_idx - self.K, DRONE

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPBTWState) -> ActionMask:
        N1 = self.n_customers + 1
        V = 2 * self.K
        mask = np.zeros(N1 * V, dtype=bool)

        for v_idx in range(V):
            k, vtype = self.vehicle_fleet_type(v_idx)
            if vtype == TRUCK:
                for j in range(1, N1):
                    if self._truck_feasible(state, k, j):
                        mask[self.encode_action(j, v_idx)] = True
                if self._truck_return_feasible(state, k):
                    mask[self.encode_action(DEPOT, v_idx)] = True
            else:
                if state.drone_active[k]:
                    for j in range(1, N1):
                        if self._drone_extend_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True
                    for land in self._landing_nodes(state, k):
                        if self._drone_land_feasible(state, k, land):
                            mask[self.encode_action(land, v_idx)] = True
                else:
                    for j in range(1, N1):
                        if self._drone_launch_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True

        return ActionMask.from_bool_array(mask)

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    def _landing_nodes(self, state: VRPBTWState, k: int) -> List[int]:
        """
        Return every node the truck visited strictly after the drone's launch
        node, plus the depot.  Landing at the launch node itself is forbidden.

        The truck route is searched from the end to find the most recent
        occurrence of drone_launch_node[k], then all subsequent entries
        (including DEPOT if the truck has returned) are valid candidates.
        """
        launch_node = int(state.drone_launch_node[k])
        route = state.truck_routes[k]

        # Find the last position of the launch node in the truck route
        launch_idx = -1
        for i in range(len(route) - 1, -1, -1):
            if route[i] == launch_node:
                launch_idx = i
                break

        # All nodes the truck visited after the launch node
        after_launch: set = (
            set(route[launch_idx + 1 :]) if launch_idx >= 0 else set(route)
        )

        # DEPOT is always reachable as a terminal landing point
        after_launch.add(DEPOT)

        # Landing at the launch node itself is forbidden
        after_launch.discard(launch_node)

        return list(after_launch)

    def _phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        phase = int(state.truck_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if j == DEPOT or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        new_load = state.truck_load[k] - self.demands[j]
        if new_load < 0 or new_load > self.Q_t:
            return False
        from_node = int(state.truck_node[k])
        arrive = state.truck_time[k] + self.manhattan_dist[from_node, j] / self.v_t
        service_end = max(arrive, self.tw_open[j]) + self.service_times[j]
        if service_end > self.tw_close[j]:
            return False
        return service_end + self.manhattan_dist[j, DEPOT] / self.v_t <= self.T_max

    def _truck_return_feasible(self, state: VRPBTWState, k: int) -> bool:
        from_node = int(state.truck_node[k])
        if from_node == DEPOT:
            return False
        arrive = state.truck_time[k] + self.manhattan_dist[from_node, DEPOT] / self.v_t
        return arrive <= self.T_max

    def _elapsed_trip_time(self, state: VRPBTWState, k: int) -> float:
        return float(state.drone_time[k] - state.drone_launch_time[k])

    def _min_return_time(self, state: VRPBTWState, k: int, from_node: int) -> float:
        times = [self.euclidean_dist[from_node, DEPOT] / self.v_d]
        t_node = int(state.truck_node[k])
        if t_node != DEPOT:
            times.append(self.euclidean_dist[from_node, t_node] / self.v_d)
        return float(min(times))

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        new_load = state.drone_load[k] - self.demands[j]
        if new_load < 0 or new_load > self.Q_d:
            return False
        drone_at = int(state.drone_node[k])
        truck_at = int(state.truck_node[k])
        if drone_at != DEPOT and drone_at != truck_at:
            return False
        t_out = self.euclidean_dist[drone_at, j] / self.v_d
        depart_t = state.drone_time[k] + self.launch_time
        arrive_j = depart_t + t_out
        service_end = max(arrive_j, self.tw_open[j]) + self.service_times[j]
        if service_end > self.tw_close[j]:
            return False
        t_back = self._min_return_time(state, k, j)
        return self.launch_time + t_out + t_back <= self.t_max

    def _drone_extend_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if not state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        new_load = state.drone_load[k] - self.demands[j]
        if new_load < 0 or new_load > self.Q_d:
            return False
        from_node = int(state.drone_node[k])
        t_to_j = self.euclidean_dist[from_node, j] / self.v_d
        arrive_j = state.drone_time[k] + t_to_j
        service_end = max(arrive_j, self.tw_open[j]) + self.service_times[j]
        if service_end > self.tw_close[j]:
            return False
        t_back = self._min_return_time(state, k, j)
        elapsed = self._elapsed_trip_time(state, k)
        return elapsed + t_to_j + t_back <= self.t_max

    def _drone_land_feasible(self, state: VRPBTWState, k: int, land: int) -> bool:
        if not state.drone_active[k]:
            return False
        from_node = int(state.drone_node[k])
        t_back = self.euclidean_dist[from_node, land] / self.v_d
        elapsed = self._elapsed_trip_time(state, k)
        if elapsed + t_back > self.t_max:
            return False
        return state.drone_time[k] + t_back + self.land_time <= self.T_max

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        node, v_idx = self.decode_action(action)
        k, vtype = self.vehicle_fleet_type(v_idx)

        # Snapshot objective before transition
        served_before = int(state.served[1:].sum())
        obj_before = self._compute_objective(state.current_cost, served_before)

        state = _copy_state(state)

        if vtype == TRUCK:
            self._apply_truck(state, k, node)
        else:
            if state.drone_active[k]:
                landing = self._landing_nodes(state, k)
                if node in landing:
                    self._apply_drone_land(state, k, node)
                else:
                    self._apply_drone_extend(state, k, node)
            else:
                self._apply_drone_launch(state, k, node)

        terminated = self._is_terminated(state)
        infeasible_end = False

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.action_space_size)
        )
        if not terminated and next_mask.is_empty():
            terminated = True
            infeasible_end = True

        # ── Reward: potential-based shaping with new objective ──
        if terminated and infeasible_end:
            reward = -1e6
        else:
            served_after = int(state.served[1:].sum())
            obj_after = self._compute_objective(state.current_cost, served_after)
            reward = -(obj_after - obj_before)

        return StepResult(
            next_state=state,
            reward=reward,
            terminated=terminated,
            truncated=False,
            action_mask=next_mask,
            info={
                "node": node,
                "fleet": k,
                "vehicle": "truck" if vtype == TRUCK else "drone",
                "served_count": int(state.served[1:].sum()),
                "current_cost": state.current_cost,
                "current_max_tard": state.current_max_tard,
            },
        )

    # ------------------------------------------------------------------
    # Transition helpers  (update state in-place, no return value)
    # ------------------------------------------------------------------

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.truck_node[k])
        dist = self.manhattan_dist[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        serve_start = max(arrive, self.tw_open[j]) if j != DEPOT else arrive
        tardiness = max(arrive - self.tw_close[j], 0.0) if j != DEPOT else 0.0

        state.truck_time[k] = serve_start + self.service_times[j]
        state.truck_node[k] = j
        state.truck_load[k] -= self.demands[j]
        state.truck_routes[k].append(j)
        if j != DEPOT:
            state.served[j] = True
            self._update_phase(state, k)

        state.current_cost += self.c_t * dist
        if j != DEPOT:
            state.current_max_tard = max(state.current_max_tard, tardiness)

    def _apply_drone_launch(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, j]
        depart_t = state.drone_time[k] + self.launch_time
        arrive_j = depart_t + dist / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)
        launch_node = int(state.truck_node[k])
        state.drone_launch_node[k] = launch_node

        state.drone_launch_time[k] = state.drone_time[k]
        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= self.demands[j]
        state.drone_active[k] = True
        state.served[j] = True
        self._update_phase(state, k)

        state.drone_route_nodes[k].extend([launch_node, j])
        state.drone_route_mask[k].extend([0, 1])

        state.current_cost += self.c_d * dist
        state.current_max_tard = max(state.current_max_tard, tardiness)

    def _apply_drone_extend(self, state: VRPBTWState, k: int, j: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, j]
        arrive_j = state.drone_time[k] + dist / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)

        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= self.demands[j]
        state.served[j] = True
        self._update_phase(state, k)

        state.drone_route_nodes[k].append(j)
        state.drone_route_mask[k].append(1)

        state.current_cost += self.c_d * dist
        state.current_max_tard = max(state.current_max_tard, tardiness)

    def _apply_drone_land(self, state: VRPBTWState, k: int, land: int) -> None:
        from_node = int(state.drone_node[k])
        dist = self.euclidean_dist[from_node, land]
        arrive = state.drone_time[k] + dist / self.v_d + self.land_time
        if land == int(state.truck_node[k]):
            arrive = max(arrive, state.truck_time[k])

        state.drone_time[k] = arrive
        state.drone_node[k] = land
        state.drone_active[k] = False
        state.drone_load[k] = self.Q_d
        state.drone_launch_time[k] = arrive

        state.drone_route_nodes[k].append(land)
        state.drone_route_mask[k].append(0)

        # Landing has no tardiness — cost only
        state.current_cost += self.c_d * dist

    # ------------------------------------------------------------------
    # Phase update
    # ------------------------------------------------------------------

    def _update_phase(self, state: VRPBTWState, k: int) -> None:
        if (
            state.truck_phase[k] == 0
            and len(self._linehaul_idx) > 0
            and state.served[self._linehaul_idx].all()
        ):
            state.truck_phase[k] = 1

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_terminated(self, state: VRPBTWState) -> bool:
        return bool(state.served[1:].all()) and not state.drone_active.any()

    # ------------------------------------------------------------------
    # state_to_obs
    # ------------------------------------------------------------------

    """
    All normalisation and static graph construction happens here, once per
    step, before anything enters the network.  The policy network receives
    only plain normalised numpy arrays — no scalar parameters, no raw coords.

    New obs dict keys:
    node_features     (N+1, 5)   normalised  [x, y, demand, tw_open, tw_close]
    vehicle_features  (2K,  5)   normalised  [x, y, rem_load, tw_open, tw_close]
    truck_edge_index  (2, E)     int32        E = (N+1)*N  all ordered pairs i≠j
    truck_edge_attr   (E,  2)    float32      [cost, time]  normalised
    drone_edge_index  (2, E)     int32
    drone_edge_attr   (E,  2)    float32      [cost, time]  normalised

    Normalisation rules
    -------------------
    Universal spatial normalizer: max_dist = 2 * max_coord (longest Manhattan distance)
    Universal time normalizer: T_scale = max_dist / min(truck_speed, drone_speed) (slowest vehicle)
    Universal capacity normalizer: max_capacity = max(Q_t, Q_d)

    x, y          / max_dist
    demand        / max_capacity
    tw_open/close / T_scale
    rem_load      / max_capacity  (both trucks and drones)
    vehicle tw_open, tw_close  / T_scale   (see details below)

    edge cost   = dist * cost_unit / (max_dist * max(c_t, c_d))
                    manhattan dist * c_t  for truck edges
                    euclidean dist * c_d  for drone edges
    edge time   = dist / max_dist / (speed * T_scale)
                    manhattan / max_dist / (v_t * T_scale)  for truck edges
                    euclidean / max_dist / (v_d * T_scale)  for drone edges

    Vehicle feature semantics
    -------------------------
    Truck k:
        tw_open  = truck_time[k] / T_scale   (earliest next departure = now)
        tw_close = T_max / T_scale            (system deadline)

    Drone k  (NOT on trip, drone_active[k] == False):
        tw_open  = truck_time[k] / T_max   (drone boards when truck arrives)
        tw_close = (drone_launch_time[k] + t_max) / T_max

    Drone k  (ON trip, drone_active[k] == True):
        tw_open  = drone_time[k] / T_max   (earliest next departure from current node)
        tw_close = (drone_launch_time[k] + t_max) / T_max

    The static graph is rebuilt from scratch each call using the instance's
    distance matrices, which are already stored on self after encode_instance.
    This is O(N^2) numpy work — fast enough for N<=50.
    """

    def state_to_obs(self, state) -> dict:
        N1 = self.n_customers + 1
        # Universal spatial normalizer: longest possible distance (Manhattan, corner to corner)
        max_dist = (
            2.0 * self.max_coord
        )  # longest distance in grid: (0,0) to (max_coord, max_coord)

        # Normalize time by the longest possible distance / slowest vehicle speed
        # This puts the entire temporal range in [0, 1] and is consistent across instances
        T = max_dist / min(self.v_t, self.v_d)  # time to traverse longest path at slowest speed

        mc = max(self.c_t, self.c_d) + 1e-8  # max cost unit
        norm_cost_denom = max_dist * mc  # cost normalisation denominator
        max_capacity = max(self.Q_t, self.Q_d) + 1e-8  # universal load normalizer

        # ── Node features ────────────────────────────────────────────────
        # Demand is zeroed for served nodes so the network can distinguish
        # visited (demand=0) from unvisited (demand≠0) without an extra feature.
        # Served linehaul/backhaul nodes also leave the l_idx/b_idx index sets
        # in NodeEncoder, stopping them from influencing routing cross-attention.
        effective_demand = np.where(state.served, 0.0, self.demands).astype(np.float32)
        node_features = np.stack(
            [
                self.coords[:, 0] / max_dist,
                self.coords[:, 1] / max_dist,
                effective_demand / max_capacity,
                self.tw_open / T,
                self.tw_close / T,
            ],
            axis=1,
        ).astype(np.float32)  # (N+1, 5)

        # ── Vehicle features ─────────────────────────────────────────────
        truck_rows = []
        drone_rows = []
        for k in range(self.K):
            # Truck k
            tx, ty = self.coords[state.truck_node[k]]
            t_rem = float(state.truck_load[k]) / max_capacity
            t_open = float(state.truck_time[k]) / T
            t_close = self.T_max / T  # system deadline normalized by time scale
            truck_rows.append(
                np.array(
                    [tx / max_dist, ty / max_dist, t_rem, t_open, t_close],
                    dtype=np.float32,
                )
            )

            # Drone k
            dx, dy = self.coords[state.drone_node[k]]
            d_rem = float(state.drone_load[k]) / max_capacity
            d_close = float(state.drone_launch_time[k] + self.t_max) / T
            if state.drone_active[k]:
                d_open = float(state.drone_time[k]) / T  # already flying
            else:
                d_open = float(state.truck_time[k]) / T  # boards when truck arrives
            drone_rows.append(
                np.array(
                    [dx / max_dist, dy / max_dist, d_rem, d_open, d_close],
                    dtype=np.float32,
                )
            )

        # Vehicle index order matches encode_action()/vehicle_fleet_type():
        # [truck_0..truck_{K-1}, drone_0..drone_{K-1}]
        vehicle_features = np.stack(truck_rows + drone_rows, axis=0)  # (2K, 5)

        # ── Candidate edges: all ordered (i, j), i != j ─────────────────────
        src, dst = np.meshgrid(np.arange(N1), np.arange(N1), indexing="ij")
        src, dst = src.ravel(), dst.ravel()
        valid = src != dst
        src, dst = src[valid], dst[valid]

        man = self.manhattan_dist[src, dst]
        euc = self.euclidean_dist[src, dst]

        truck_cost = (man * self.c_t) / norm_cost_denom
        truck_time = (man / max_dist) / (self.v_t * T + 1e-8)
        drone_cost = (euc * self.c_d) / norm_cost_denom
        drone_time = (euc / max_dist) / (self.v_d * T + 1e-8)

        truck_edge_attr_all = np.stack([truck_cost, truck_time], axis=1).astype(
            np.float32
        )
        drone_edge_attr_all = np.stack([drone_cost, drone_time], axis=1).astype(
            np.float32
        )

        # ── Sparse edge sets ──────────────────────────────────────────────
        #
        # Truck subgraph — keep endpoints that are routing-relevant for the truck:
        #   • unserved customers        (future truck targets)
        #   • depot                     (always kept — trucks return, drones land)
        #   • current truck node(s)     (planning origin)
        #   • landing candidates        (post-launch truck nodes the active drone
        #                                can rendezvous with; kept so the GNN
        #                                propagates structure around those sites)
        #
        # Drone subgraph — same keep set as truck, plus the current drone node:
        #   • current drone node(s)     (active: extend-trip or land decisions;
        #                                inactive: next-launch origin)
        #
        # Edges where EITHER endpoint falls outside the keep set are dropped.
        # Nodes left with no edges in either subgraph still participate in the
        # attention encoders (NodeEncoder / VehicleEncoder) but receive no GNN
        # message aggregation — their Z_graph embedding is feature-only.

        keep_truck = ~state.served.copy()  # unserved customers
        keep_truck[DEPOT] = True
        for k in range(self.K):
            keep_truck[int(state.truck_node[k])] = True  # current truck pos
            if state.drone_active[k]:
                for lc in self._landing_nodes(state, k):  # landing candidates
                    keep_truck[lc] = True

        keep_drone = keep_truck.copy()
        for k in range(self.K):
            keep_drone[int(state.drone_node[k])] = True  # current drone pos

        t_mask = keep_truck[src] & keep_truck[dst]
        d_mask = keep_drone[src] & keep_drone[dst]

        truck_edge_index = np.stack([src[t_mask], dst[t_mask]], axis=0).astype(
            np.int32
        )  # (2, E_t)
        drone_edge_index = np.stack([src[d_mask], dst[d_mask]], axis=0).astype(
            np.int32
        )  # (2, E_d)

        return {
            "node_features": node_features,  # (N+1, 5)
            "vehicle_features": vehicle_features,  # (2K,  5)
            "truck_edge_index": truck_edge_index,  # (2,   E_t)
            "truck_edge_attr": truck_edge_attr_all[t_mask],  # (E_t, 2)
            "drone_edge_index": drone_edge_index,  # (2,   E_d)
            "drone_edge_attr": drone_edge_attr_all[d_mask],  # (E_d, 2)
        }

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> Tuple[float, int]:
        """
        Evaluate solution metrics by replaying stored routes.

        Returns:
            (total_cost, served_count) - raw solution properties
        """
        total_cost = 0.0

        for k in range(self.K):
            t, prev = 0.0, DEPOT
            for j in state.truck_routes[k]:
                dist = self.manhattan_dist[prev, j]
                t += dist / self.v_t
                if j != DEPOT:
                    t = max(t, self.tw_open[j]) + self.service_times[j]
                total_cost += self.c_t * dist
                prev = j

            t_d, prev_d, in_trip = 0.0, DEPOT, False
            for node, is_cust in zip(
                state.drone_route_nodes[k], state.drone_route_mask[k]
            ):
                dist = self.euclidean_dist[prev_d, node]
                total_cost += self.c_d * dist
                if not in_trip and not is_cust:
                    t_d = max(t_d + dist / self.v_d, 0.0)
                elif not in_trip and is_cust:
                    t_d += self.launch_time + dist / self.v_d
                    t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                    in_trip = True
                elif in_trip and is_cust:
                    t_d += dist / self.v_d
                    t_d = max(t_d, self.tw_open[node]) + self.service_times[node]
                else:
                    t_d += dist / self.v_d + self.land_time
                    in_trip = False
                prev_d = node

        served_count = int(state.served[1:].sum())
        return total_cost, served_count


    def is_complete(self, state: VRPBTWState) -> bool:
        return self._is_terminated(state)

    def decode_solution(self, state: VRPBTWState) -> Solution:
        cost, served_count = self.evaluate(state)
        obj = self._compute_objective(cost, served_count)
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=obj,
            metadata={
                "total_cost": cost,
                "served_count": served_count,
                "n_customers": self.n_customers,
                "truck_routes": [list(r) for r in state.truck_routes],
                "drone_route_nodes": [list(r) for r in state.drone_route_nodes],
                "drone_route_mask": [list(m) for m in state.drone_route_mask],
                "unserved": int((~state.served[1:]).sum()),
            },
        )


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return (self.n_customers + 1) * 2 * self.K

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.n_customers + 1, NODE_FEAT_DIM)

    @property
    def n_vehicles(self) -> int:
        return 2 * self.K

    @property
    def linehaul_indices(self) -> np.ndarray:
        return self._linehaul_idx

    @property
    def backhaul_indices(self) -> np.ndarray:
        return self._backhaul_idx

    def get_candidate_starts(self) -> List[Tuple[Any, Dict, Dict]]:
        """
        Return candidate starting states for POMO.

        For each linehaul customer j, create a starting state where truck 0 (fleet index 0)
        has just been dispatched to j. This ensures:
        - All candidates are feasible (truck 0 can reach each linehaul customer from depot)
        - Diversity in starting configurations
        - Truck 0 always takes the first action (to a linehaul customer)

        Returns
        -------
        list of (state, obs, info) — one per linehaul customer. If no linehaul customers
        are feasible, returns a single tuple with the initial state.
        """
        candidates = []
        s0 = self.initial_state()

        for j in self._linehaul_idx:
            action = self.encode_action(int(j), 0)
            result = self.apply_action(s0, action)

            if not result.terminated and not result.action_mask.is_empty():
                obs = self.state_to_obs(result.next_state)
                info = {
                    "action_mask": result.action_mask.mask,
                    "feasible_actions": result.action_mask.action_indices,
                }
                candidates.append((result.next_state, obs, info))

        if not candidates:
            obs = self.state_to_obs(s0)
            mask = self.get_action_mask(s0)
            info = {
                "action_mask": mask.mask,
                "feasible_actions": mask.action_indices,
            }
            candidates.append((s0, obs, info))

        return candidates
