"""
problems/vrpbtw/problem.py
--------------------------
VRPBTW with heterogeneous truck-drone fleets.

Key design decisions
--------------------
1. Signed demand encoding
   demand > 0  -> linehaul (delivery)
   demand < 0  -> backhaul (pickup)
   demand = 0  -> depot
   No separate type field needed — sign is the type signal.
   Normalised by Q_t so the range is roughly [-1, 1].

2. Drone airborne state (Approach 1 — deferred landing)
   When a drone is dispatched it enters an AIRBORNE state.
   The landing node is decided at a LATER step as a separate
   special action, once the truck has moved to a feasible
   rendezvous position.
   Fields:
     drone_active:            (K,) bool  — is drone airborne?
     drone_airborne_customer: (K,) int   — customer being served (-1 idle)
     drone_rendezvous:        (K,) int   — planned landing node  (-1 none)

3. Observation is a DICT with two arrays
   "node_features"    : (N+1, NODE_FEAT_DIM=9)   — for the encoder
   "vehicle_features" : (2K,  VEH_FEAT_DIM=7)    — for the decoder context

4. Hierarchical action helpers
   encode_action(fleet, vehicle, node) -> flat int
   decode_action(flat int) -> (fleet, vehicle, node)
   vehicle_index(fleet, vehicle) -> global vehicle id 0..2K-1

5. Node features  (NODE_FEAT_DIM = 9)
   0: x / max_coord
   1: y / max_coord
   2: demand / Q_t          signed: >0 linehaul, <0 backhaul, 0 depot
   3: e_i / T_max
   4: l_i / T_max
   5: (l_i-e_i) / T_max     slack width
   6: s_i / T_max            service time
   7: served (float)         dynamic
   8: lambda                 scalarisation weight

6. Vehicle features  (VEH_FEAT_DIM = 7)
   Ordered: truck_0, drone_0, truck_1, drone_1, ...
   0: current_node / N
   1: remaining_cap / Q_v
   2: current_time / T_max
   3: battery fraction
   4: is_drone (0/1)
   5: fleet_id / K
   6: phase (0=linehaul, 1=backhaul)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import Problem, ActionMask, StepResult
from core.solution import Solution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 9
VEH_FEAT_DIM = 7
TRUCK = 0
DRONE = 1
DEPOT = 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # Truck (one per fleet)
    truck_node: np.ndarray  # (K,)   int  current node index
    truck_pos: np.ndarray  # (K, 2) 2D position
    truck_load: np.ndarray  # (K,)   remaining capacity
    truck_time: np.ndarray  # (K,)   earliest available time
    truck_dist: np.ndarray  # (K,)   accumulated distance
    truck_phase: np.ndarray  # (K,)   int 0=linehaul 1=backhaul
    truck_route: List[List[int]]

    # Drone (one per fleet)
    drone_node: np.ndarray  # (K,)   last known node
    drone_pos: np.ndarray  # (K, 2) position when idle
    drone_load: np.ndarray  # (K,)   remaining capacity
    drone_time: np.ndarray  # (K,)   earliest available time
    drone_dist: np.ndarray  # (K,)   accumulated distance
    drone_battery: np.ndarray  # (K,)   remaining battery
    drone_active: np.ndarray  # (K,)   bool airborne?
    drone_airborne_customer: np.ndarray  # (K,)   int customer being served (-1 idle)
    drone_rendezvous: np.ndarray  # (K,)   int planned landing node (-1 none)
    drone_route: List[List[int]]

    # Global
    served: np.ndarray  # (N+1,) bool
    step: int
    max_tardiness: float


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class VRPBTWProblem(Problem):
    def __init__(self, n_customers: int = 10, n_fleets: int = 2):
        super().__init__(name="VRPBTW")
        self.n_customers = n_customers
        self.n_fleets = n_fleets

        self.coords: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.tw_open: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_close: np.ndarray = np.zeros(1, dtype=np.float32)
        self.demands: np.ndarray = np.zeros(1, dtype=np.float32)
        self.service_times: np.ndarray = np.zeros(1, dtype=np.float32)
        self.dist_matrix: np.ndarray = np.zeros((1, 1), dtype=np.float32)

        self.K: int = n_fleets
        self.Q_t: float = 1.0
        self.Q_d: float = 1.0
        self.T_max: float = 1.0
        self.D_max: float = 1.0
        self.v_t: float = 1.0
        self.v_d: float = 2.0
        self.c_t: float = 1.0
        self.c_d: float = 0.5
        self.launch_time: float = 0.0
        self.land_time: float = 0.0
        self.service_time: float = 0.0
        self.lambda_weight: float = 1.0

        self._linehaul_indices: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_indices: np.ndarray = np.array([], dtype=np.int32)

    # ------------------------------------------------------------------
    # encode_instance
    # ------------------------------------------------------------------

    def encode_instance(self, raw_instance: Dict) -> None:
        depot = np.array(raw_instance["depot"], dtype=np.float32)
        customers = np.array(raw_instance["customers"], dtype=np.float32)

        self.n_customers = len(customers)
        self.K = int(raw_instance["n_fleets"])

        coords_all = np.vstack([depot, customers[:, :2]])
        tw_open_all = np.concatenate([[0.0], customers[:, 2]])
        tw_close_all = np.concatenate(
            [[raw_instance["system_duration"]], customers[:, 3]]
        )
        demands_all = np.concatenate([[0.0], customers[:, 4]])

        self.coords = coords_all.astype(np.float32)
        self.tw_open = tw_open_all.astype(np.float32)
        self.tw_close = tw_close_all.astype(np.float32)
        self.demands = demands_all.astype(np.float32)

        svc = float(raw_instance.get("service_time", 0.0))
        self.service_times = np.full(self.n_customers + 1, svc, dtype=np.float32)
        self.service_times[DEPOT] = 0.0

        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist_matrix = np.sqrt((diff**2).sum(-1)).astype(np.float32)

        self.Q_t = float(raw_instance["truck_capacity"])
        self.Q_d = float(raw_instance["drone_capacity"])
        self.T_max = float(raw_instance["system_duration"])
        self.D_max = float(raw_instance["drone_trip_duration"])
        self.v_t = float(raw_instance["truck_speed"])
        self.v_d = float(raw_instance["drone_speed"])
        self.c_t = float(raw_instance["truck_cost"])
        self.c_d = float(raw_instance["drone_cost"])
        self.launch_time = float(raw_instance["launch_time"])
        self.land_time = float(raw_instance["land_time"])
        self.service_time = float(raw_instance.get("service_time", 0.0))
        self.lambda_weight = float(raw_instance.get("lambda_weight", 1.0))

        cust_demands = self.demands[1:]
        self._linehaul_indices = (np.where(cust_demands > 0)[0] + 1).astype(np.int32)
        self._backhaul_indices = (np.where(cust_demands < 0)[0] + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # initial_state
    # ------------------------------------------------------------------

    def initial_state(self) -> VRPBTWState:
        K = self.K
        depot = self.coords[DEPOT]
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True

        return VRPBTWState(
            truck_node=np.zeros(K, dtype=np.int32),
            truck_pos=np.tile(depot, (K, 1)).astype(np.float32),
            truck_load=np.full(K, self.Q_t, dtype=np.float32),
            truck_time=np.zeros(K, dtype=np.float32),
            truck_dist=np.zeros(K, dtype=np.float32),
            truck_phase=np.zeros(K, dtype=np.int32),
            truck_route=[[DEPOT] for _ in range(K)],
            drone_node=np.zeros(K, dtype=np.int32),
            drone_pos=np.tile(depot, (K, 1)).astype(np.float32),
            drone_load=np.full(K, self.Q_d, dtype=np.float32),
            drone_time=np.zeros(K, dtype=np.float32),
            drone_dist=np.zeros(K, dtype=np.float32),
            drone_battery=np.full(K, self.D_max, dtype=np.float32),
            drone_active=np.zeros(K, dtype=bool),
            drone_airborne_customer=np.full(K, -1, dtype=np.int32),
            drone_rendezvous=np.full(K, -1, dtype=np.int32),
            drone_route=[[] for _ in range(K)],
            served=served,
            step=0,
            max_tardiness=0.0,
        )

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def encode_action(self, fleet: int, vehicle: int, node: int) -> int:
        N1 = self.n_customers + 1
        return fleet * (2 * N1) + vehicle * N1 + node

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        N1 = self.n_customers + 1
        fleet = action // (2 * N1)
        vehicle = (action // N1) % 2
        node = action % N1
        return fleet, vehicle, node

    def vehicle_index(self, fleet: int, vehicle: int) -> int:
        return fleet * 2 + vehicle

    # ------------------------------------------------------------------
    # get_action_mask
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPBTWState) -> ActionMask:
        N1 = self.n_customers + 1
        size = self.K * 2 * N1
        mask = np.zeros(size, dtype=bool)

        # Airborne drones take priority — must land before other moves
        airborne = np.where(state.drone_active)[0]
        if len(airborne) > 0:
            for k in airborne:
                for j in self._landing_candidates(state, k):
                    mask[self.encode_action(k, DRONE, j)] = True
            return ActionMask.from_bool_array(mask)

        for k in range(self.K):
            for v in (TRUCK, DRONE):
                for j in range(N1):
                    mask[self.encode_action(k, v, j)] = self._is_feasible(
                        state, k, v, j
                    )

        return ActionMask.from_bool_array(mask)

    def _landing_candidates(self, state: VRPBTWState, k: int) -> List[int]:
        cust = int(state.drone_airborne_customer[k])
        from_node = cust if cust >= 0 else DEPOT
        candidates = []
        for j in [DEPOT, int(state.truck_node[k])]:
            dist_back = self.dist_matrix[from_node, j]
            arrive_land = state.drone_time[k] + dist_back / self.v_d + self.land_time
            if arrive_land <= self.T_max:
                candidates.append(j)
        return candidates if candidates else [DEPOT]

    def _is_feasible(self, state: VRPBTWState, k: int, v: int, j: int) -> bool:
        if j == DEPOT:
            return True
        if state.served[j]:
            return False
        return (
            self._truck_feasible(state, k, j)
            if v == TRUCK
            else self._drone_feasible(state, k, j)
        )

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        # Phase constraint: cannot serve backhaul before all linehaul done
        if state.truck_phase[k] == 0 and self.demands[j] < 0:
            return False
        if abs(self.demands[j]) > state.truck_load[k]:
            return False
        from_node = int(state.truck_node[k])
        dist = self.dist_matrix[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        if arrive > self.tw_close[j]:
            return False
        depart = max(arrive, self.tw_open[j]) + self.service_times[j]
        if depart > self.T_max:
            return False
        if depart + self.dist_matrix[j, DEPOT] / self.v_t > self.T_max:
            return False
        return True

    def _drone_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if state.drone_active[k]:
            return False
        if not np.allclose(state.drone_pos[k], state.truck_pos[k], atol=1e-4):
            return False
        if self.demands[j] < 0:  # drones don't serve backhaul
            return False
        if abs(self.demands[j]) > state.drone_load[k]:
            return False
        from_node = int(state.truck_node[k])
        dist_out = self.dist_matrix[from_node, j]
        dist_back = self.dist_matrix[j, DEPOT]
        airborne = (dist_out + dist_back) / self.v_d
        if airborne > state.drone_battery[k]:
            return False
        depart_time = state.drone_time[k]
        arrive_j = depart_time + self.launch_time + dist_out / self.v_d
        if arrive_j > self.tw_close[j]:
            return False
        finish = (
            depart_time
            + self.launch_time
            + airborne
            + self.service_times[j]
            + self.land_time
        )
        return finish <= self.T_max

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        N1 = self.n_customers + 1
        k, v, j = self.decode_action(action)
        state = _copy_state(state)

        if state.drone_active[k] and v == DRONE:
            reward = self._apply_drone_landing(state, k, j)
        elif v == TRUCK:
            reward = self._apply_truck(state, k, j)
        else:
            reward = self._apply_drone_launch(state, k, j)

        state.step += 1

        terminated = bool(state.served[1:].all()) and not state.drone_active.any()
        if terminated:
            reward += self._terminal_reward(state)

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.K * 2 * N1)
        )

        if not terminated and next_mask.is_empty():
            terminated = True
            reward += self._terminal_reward(state)

        return StepResult(
            next_state=state,
            reward=reward,
            terminated=terminated,
            truncated=False,
            action_mask=next_mask,
            info={
                "fleet": k,
                "vehicle": "truck" if v == TRUCK else "drone",
                "node": j,
                "served_count": int(state.served[1:].sum()),
                "max_tardiness": state.max_tardiness,
            },
        )

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> float:
        from_node = int(state.truck_node[k])
        dist = self.dist_matrix[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        serve_start = max(arrive, self.tw_open[j])

        tardiness = max(serve_start - self.tw_close[j], 0.0)
        state.max_tardiness = max(state.max_tardiness, tardiness)

        state.truck_time[k] = serve_start + self.service_times[j]
        state.truck_pos[k] = self.coords[j].copy()
        state.truck_node[k] = j
        state.truck_dist[k] += dist
        state.truck_load[k] -= abs(self.demands[j])
        state.truck_route[k].append(j)

        if j != DEPOT:
            state.served[j] = True
            if (
                state.truck_phase[k] == 0
                and len(self._linehaul_indices) > 0
                and state.served[self._linehaul_indices].all()
            ):
                state.truck_phase[k] = 1

        if state.drone_rendezvous[k] == j:
            state.drone_time[k] = max(state.drone_time[k], state.truck_time[k])
            state.drone_pos[k] = self.coords[j].copy()
            state.drone_node[k] = j
            state.drone_active[k] = False
            state.drone_airborne_customer[k] = -1
            state.drone_rendezvous[k] = -1

        tw_bonus = 0.1 * max(self.tw_close[j] - arrive, 0.0) / (self.T_max + 1e-6)
        return float(-(self.c_t * dist) - self.lambda_weight * tardiness + tw_bonus)

    def _apply_drone_launch(self, state: VRPBTWState, k: int, j: int) -> float:
        from_node = int(state.truck_node[k])
        dist_out = self.dist_matrix[from_node, j]
        depart_time = state.drone_time[k]
        t_launch = depart_time + self.launch_time
        t_arrive_j = t_launch + dist_out / self.v_d
        t_serve_end = max(t_arrive_j, self.tw_open[j]) + self.service_times[j]

        tardiness = max(t_arrive_j - self.tw_close[j], 0.0)
        state.max_tardiness = max(state.max_tardiness, tardiness)

        state.drone_active[k] = True
        state.drone_airborne_customer[k] = j
        state.drone_time[k] = t_serve_end
        state.drone_dist[k] += dist_out
        state.drone_load[k] -= abs(self.demands[j])
        state.drone_battery[k] -= dist_out / self.v_d
        state.drone_route[k].append(j)

        if j != DEPOT:
            state.served[j] = True

        tw_bonus = 0.1 * max(self.tw_close[j] - t_arrive_j, 0.0) / (self.T_max + 1e-6)
        return float(-(self.c_d * dist_out) - self.lambda_weight * tardiness + tw_bonus)

    def _apply_drone_landing(self, state: VRPBTWState, k: int, j: int) -> float:
        cust = int(state.drone_airborne_customer[k])
        from_node = cust if cust >= 0 else DEPOT
        dist_back = self.dist_matrix[from_node, j]
        t_arrive = state.drone_time[k] + dist_back / self.v_d
        t_landed = t_arrive + self.land_time

        state.drone_time[k] = t_landed
        state.drone_pos[k] = self.coords[j].copy()
        state.drone_node[k] = j
        state.drone_dist[k] += dist_back
        state.drone_battery[k] -= dist_back / self.v_d
        state.drone_active[k] = False
        state.drone_airborne_customer[k] = -1
        state.drone_rendezvous[k] = -1

        return float(-(self.c_d * dist_back))

    # ------------------------------------------------------------------
    # Terminal reward
    # ------------------------------------------------------------------

    def _terminal_reward(self, state: VRPBTWState) -> float:
        unserved = int((~state.served[1:]).sum())
        base = 10.0 if unserved == 0 else -2.0 * unserved
        return float(base - self.lambda_weight * state.max_tardiness)

    # ------------------------------------------------------------------
    # state_to_obs  — returns dict
    # ------------------------------------------------------------------

    def state_to_obs(self, state: VRPBTWState) -> Dict[str, np.ndarray]:
        N1 = self.n_customers + 1
        T = self.T_max + 1e-6
        max_coord = float(self.coords.max()) + 1e-6

        tw_width = np.clip((self.tw_close - self.tw_open) / T, 0.0, 1.0)

        node_feat = np.stack(
            [
                self.coords[:, 0] / max_coord,
                self.coords[:, 1] / max_coord,
                self.demands / (self.Q_t + 1e-6),  # signed demand
                self.tw_open / T,
                self.tw_close / T,
                tw_width,
                self.service_times / T,
                state.served.astype(np.float32),
                np.full(N1, self.lambda_weight, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)  # (N+1, 9)

        veh_rows = []
        for k in range(self.K):
            veh_rows.append(
                np.array(
                    [
                        state.truck_node[k] / max(N1 - 1, 1),
                        state.truck_load[k] / (self.Q_t + 1e-6),
                        state.truck_time[k] / T,
                        1.0,
                        0.0,
                        k / max(self.K - 1, 1),
                        float(state.truck_phase[k]),
                    ],
                    dtype=np.float32,
                )
            )
            veh_rows.append(
                np.array(
                    [
                        state.drone_node[k] / max(N1 - 1, 1),
                        state.drone_load[k] / (self.Q_d + 1e-6),
                        state.drone_time[k] / T,
                        state.drone_battery[k] / (self.D_max + 1e-6),
                        1.0,
                        k / max(self.K - 1, 1),
                        float(state.truck_phase[k]),
                    ],
                    dtype=np.float32,
                )
            )

        vehicle_feat = np.stack(veh_rows, axis=0)  # (2K, 7)

        return {
            "node_features": node_feat,
            "vehicle_features": vehicle_feat,
        }

    # ------------------------------------------------------------------
    # evaluate / is_complete / properties
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> float:
        truck_cost = float((state.truck_dist * self.c_t).sum())
        drone_cost = float((state.drone_dist * self.c_d).sum())
        unserved = int((~state.served[1:]).sum())
        return -(
            truck_cost
            + drone_cost
            + self.lambda_weight * state.max_tardiness
            + unserved * 1000.0
        )

    def is_complete(self, state: VRPBTWState) -> bool:
        return bool(state.served[1:].all()) and not state.drone_active.any()

    @property
    def action_space_size(self) -> int:
        return self.K * 2 * (self.n_customers + 1)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.n_customers + 1, NODE_FEAT_DIM)

    @property
    def n_vehicles(self) -> int:
        return 2 * self.K

    @property
    def linehaul_indices(self) -> np.ndarray:
        return self._linehaul_indices

    @property
    def backhaul_indices(self) -> np.ndarray:
        return self._backhaul_indices

    # ------------------------------------------------------------------
    # decode_solution / heuristic
    # ------------------------------------------------------------------

    def decode_solution(self, state: VRPBTWState) -> Solution:
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state),
            metadata={
                "served_count": int(state.served[1:].sum()),
                "n_customers": self.n_customers,
                "truck_routes": [list(r) for r in state.truck_route],
                "drone_routes": [list(r) for r in state.drone_route],
                "truck_dist": state.truck_dist.tolist(),
                "drone_dist": state.drone_dist.tolist(),
                "truck_times": state.truck_time.tolist(),
                "drone_times": state.drone_time.tolist(),
                "max_tardiness": state.max_tardiness,
                "unserved": int((~state.served[1:]).sum()),
            },
        )

    def heuristic_solution(self) -> Optional[float]:
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        current, time, dist, load = DEPOT, 0.0, 0.0, self.Q_t
        while not served[1:].all():
            best_j, best_d = -1, float("inf")
            for j in range(1, self.n_customers + 1):
                if served[j] or abs(self.demands[j]) > load:
                    continue
                d = self.dist_matrix[current, j]
                arrive = time + d / self.v_t
                if arrive > self.tw_close[j]:
                    continue
                if d < best_d:
                    best_d, best_j = d, j
            if best_j == -1:
                break
            d = self.dist_matrix[current, best_j]
            time = (
                max(time + d / self.v_t, self.tw_open[best_j])
                + self.service_times[best_j]
            )
            dist += d
            load -= abs(self.demands[best_j])
            served[best_j] = True
            current = best_j
        dist += self.dist_matrix[current, DEPOT]
        unserved = int((~served[1:]).sum())
        return -(dist * self.c_t + unserved * 1000.0)


# ---------------------------------------------------------------------------
# State copy
# ---------------------------------------------------------------------------


def _copy_state(s: VRPBTWState) -> VRPBTWState:
    return VRPBTWState(
        truck_node=s.truck_node.copy(),
        truck_pos=s.truck_pos.copy(),
        truck_load=s.truck_load.copy(),
        truck_time=s.truck_time.copy(),
        truck_dist=s.truck_dist.copy(),
        truck_phase=s.truck_phase.copy(),
        truck_route=[list(r) for r in s.truck_route],
        drone_node=s.drone_node.copy(),
        drone_pos=s.drone_pos.copy(),
        drone_load=s.drone_load.copy(),
        drone_time=s.drone_time.copy(),
        drone_dist=s.drone_dist.copy(),
        drone_battery=s.drone_battery.copy(),
        drone_active=s.drone_active.copy(),
        drone_airborne_customer=s.drone_airborne_customer.copy(),
        drone_rendezvous=s.drone_rendezvous.copy(),
        drone_route=[list(r) for r in s.drone_route],
        served=s.served.copy(),
        step=s.step,
        max_tardiness=s.max_tardiness,
    )


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def generate_vrpbtw(
    n_customers: int = 10,
    n_fleets: int = 2,
    grid_size: float = 100.0,
    linehaul_ratio: float = 0.5,
    lambda_weight: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    depot_xy = (grid_size / 2.0, grid_size / 2.0)
    coords = rng.uniform(0.0, grid_size, (n_customers, 2))

    n_linehaul = max(1, int(n_customers * linehaul_ratio))
    n_backhaul = n_customers - n_linehaul

    demands = np.concatenate(
        [
            rng.uniform(1.0, 10.0, n_linehaul),  # positive linehaul
            -rng.uniform(1.0, 10.0, n_backhaul),  # negative backhaul
        ]
    )

    idx = rng.permutation(n_customers)
    coords = coords[idx]
    demands = demands[idx]

    depot_arr = np.array(depot_xy)
    dist_depot = np.linalg.norm(coords - depot_arr, axis=1)
    earliest_arr = dist_depot / 1.0

    tw_open = np.maximum(0.0, earliest_arr - rng.uniform(5.0, 15.0, n_customers))
    tw_close = earliest_arr + rng.uniform(20.0, 50.0, n_customers)

    system_duration = float(tw_close.max() + 30.0)
    drone_trip_duration = grid_size * 0.6

    customers = np.column_stack([coords, tw_open, tw_close, demands]).tolist()

    return {
        "depot": list(depot_xy),
        "customers": customers,
        "n_fleets": n_fleets,
        "truck_capacity": 50.0,
        "drone_capacity": 15.0,
        "system_duration": system_duration,
        "drone_trip_duration": drone_trip_duration,
        "truck_speed": 1.0,
        "drone_speed": 2.0,
        "truck_cost": 1.0,
        "drone_cost": 0.5,
        "launch_time": 2.0,
        "land_time": 3.0,
        "service_time": 5.0,
        "lambda_weight": lambda_weight,
    }
