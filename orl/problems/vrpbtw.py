"""
problems/vrpbtw.py
------------------
VRPBTW with heterogeneous truck-drone fleets.

Key design decisions
--------------------
1. Signed demand encoding
   demand > 0  -> linehaul (delivery)
   demand < 0  -> backhaul (pickup)
   demand = 0  -> depot

2. Multi-stop drone trips
   A single drone trip: launch_node -> c1 -> c2 -> ... -> cn -> land_node
   - launch_node / land_node: depot (0) or truck node of same fleet
   - trip budget D_max resets on every landing
   - drone_load resets on every landing (Q_d per trip)
   - linehaul drone trips only in truck linehaul phase
   - backhaul drone trips only in truck backhaul phase

3. Phase constraint
   truck_phase 0 = linehaul: truck and drone can only serve demand > 0
   truck_phase 1 = backhaul: truck and drone can only serve demand < 0
   Phase flips when all linehaul customers of that fleet are served.

4. Bidirectional action MDP
   Action = (node, vehicle) selected jointly.
   Upper policy selects node (customer or landing node).
   Lower policy selects vehicle given selected node.

5. State routes = GNN input
   truck_routes:      List[List[int]]  node sequences (no leading depot)
   drone_route_nodes: List[List[int]]  waypoint sequences (no leading depot)
   drone_route_mask:  List[List[int]]  0=truck-node waypoint, 1=drone-served customer
   Trailing depot (0) appended only when vehicle actually returns.

6. Graph edges stored incrementally in state
   edge_index: (2, E) src/dst in original customer index space
   edge_attr:  (E, 6) [edge_type, travel_time, travel_dist,
                        depart_time, arrive_time, tardiness]
   edge_fleet: (E,)   fleet id of owning edge

Node features (NODE_FEAT_DIM = 9)
   0: x / max_coord
   1: y / max_coord
   2: demand / Q_t          signed
   3: tw_open / T_max
   4: tw_close / T_max
   5: (tw_close - tw_open) / T_max   slack
   6: service_time / T_max
   7: served (float)
   8: lambda_weight

Vehicle features (VEH_FEAT_DIM = 6)  -- fed as property MLP, NOT as graph node
   Truck: [current_node/N, remaining_load/Q_t, current_time/T_max,
           phase(0/1),      0.0,                0.0             ]
   Drone: [current_node/N, remaining_load/Q_d, current_time/T_max,
           0.0,             remaining_dist/D_max, float(active) ]
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
VEH_FEAT_DIM = 6
EDGE_FEAT_DIM = 6
TRUCK = 0
DRONE = 1
DEPOT = 0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # --- per fleet dynamic scalars ---
    truck_node: np.ndarray  # (K,) int   current node
    truck_time: np.ndarray  # (K,) float earliest available time
    truck_load: np.ndarray  # (K,) float remaining capacity
    truck_phase: np.ndarray  # (K,) int   0=linehaul 1=backhaul

    drone_node: np.ndarray  # (K,) int   current node (last known)
    drone_time: np.ndarray  # (K,) float earliest available time
    drone_load: np.ndarray  # (K,) float remaining trip capacity
    drone_dist: np.ndarray  # (K,) float accumulated dist this trip
    drone_active: np.ndarray  # (K,) bool  airborne

    # --- global ---
    served: np.ndarray  # (N+1,) bool  index 0 = depot (always True)

    # --- fleet routes (GNN input + logging) ---
    truck_routes: List[List[int]]  # (K,) node sequences, no leading depot
    drone_route_nodes: List[List[int]]  # (K,) waypoint sequences, no leading depot
    drone_route_mask: List[List[int]]  # (K,) 0=truck-node 1=drone-customer

    # --- graph edges (built incrementally, fed to GNN) ---
    edge_index: np.ndarray  # (2, E) int
    edge_attr: np.ndarray  # (E, EDGE_FEAT_DIM) float
    edge_fleet: np.ndarray  # (E,) int


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class VRPBTWProblem(Problem):
    def __init__(self, n_customers: int = 10, n_fleets: int = 2):
        super().__init__(name="VRPBTW")
        self.n_customers = n_customers
        self.n_fleets = n_fleets

        # filled by encode_instance
        self.coords: np.ndarray = np.zeros((1, 2), dtype=np.float32)
        self.demands: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_open: np.ndarray = np.zeros(1, dtype=np.float32)
        self.tw_close: np.ndarray = np.zeros(1, dtype=np.float32)
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
        self.lambda_weight: float = 1.0

        self._linehaul_idx: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_idx: np.ndarray = np.array([], dtype=np.int32)

    # ------------------------------------------------------------------
    # encode_instance
    # ------------------------------------------------------------------

    def encode_instance(self, raw: Dict) -> None:
        depot = np.array(raw["depot"], dtype=np.float32)
        customers = np.array(raw["customers"], dtype=np.float32)

        self.n_customers = len(customers)
        self.K = int(raw["n_fleets"])

        coords_all = np.vstack([depot, customers[:, :2]])
        tw_open_all = np.concatenate([[0.0], customers[:, 2]])
        tw_close_all = np.concatenate(
            [[float(raw["system_duration"])], customers[:, 3]]
        )
        demands_all = np.concatenate([[0.0], customers[:, 4]])
        svc = float(raw.get("service_time", 0.0))
        svc_all = np.full(self.n_customers + 1, svc, dtype=np.float32)
        svc_all[DEPOT] = 0.0

        self.coords = coords_all.astype(np.float32)
        self.tw_open = tw_open_all.astype(np.float32)
        self.tw_close = tw_close_all.astype(np.float32)
        self.demands = demands_all.astype(np.float32)
        self.service_times = svc_all

        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist_matrix = np.sqrt((diff**2).sum(-1)).astype(np.float32)

        self.Q_t = float(raw["truck_capacity"])
        self.Q_d = float(raw["drone_capacity"])
        self.T_max = float(raw["system_duration"])
        self.D_max = float(raw["drone_trip_duration"])
        self.v_t = float(raw["truck_speed"])
        self.v_d = float(raw["drone_speed"])
        self.c_t = float(raw["truck_cost"])
        self.c_d = float(raw["drone_cost"])
        self.launch_time = float(raw.get("launch_time", 0.0))
        self.land_time = float(raw.get("land_time", 0.0))
        self.lambda_weight = float(raw.get("lambda_weight", 1.0))

        cust_d = self.demands[1:]
        self._linehaul_idx = (np.where(cust_d > 0)[0] + 1).astype(np.int32)
        self._backhaul_idx = (np.where(cust_d < 0)[0] + 1).astype(np.int32)

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
            drone_dist=np.zeros(K, dtype=np.float32),
            drone_active=np.zeros(K, dtype=bool),
            served=served,
            truck_routes=[[] for _ in range(K)],
            drone_route_nodes=[[] for _ in range(K)],
            drone_route_mask=[[] for _ in range(K)],
            edge_index=np.zeros((2, 0), dtype=np.int32),
            edge_attr=np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
            edge_fleet=np.zeros(0, dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def encode_action(self, node: int, vehicle_idx: int) -> int:
        """
        vehicle_idx: 0..K-1 = truck k, K..2K-1 = drone k
        flat = node * 2K + vehicle_idx
        """
        return node * (2 * self.K) + vehicle_idx

    def decode_action(self, action: int) -> Tuple[int, int]:
        """Returns (node, vehicle_idx)."""
        return action // (2 * self.K), action % (2 * self.K)

    def vehicle_fleet(self, vehicle_idx: int) -> Tuple[int, int]:
        """Returns (fleet_k, vehicle_type) from vehicle_idx."""
        if vehicle_idx < self.K:
            return vehicle_idx, TRUCK
        return vehicle_idx - self.K, DRONE

    # ------------------------------------------------------------------
    # get_action_mask
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPBTWState) -> ActionMask:
        N1 = self.n_customers + 1
        V = 2 * self.K
        size = N1 * V
        mask = np.zeros(size, dtype=bool)

        for v_idx in range(V):
            k, vtype = self.vehicle_fleet(v_idx)

            if vtype == TRUCK:
                # truck can visit unserved customers (phase-gated)
                for j in range(1, N1):
                    if self._truck_feasible(state, k, j):
                        mask[self.encode_action(j, v_idx)] = True
                # truck can return to depot if at least one customer served
                if any(len(r) > 0 for r in state.truck_routes):
                    if self._truck_return_feasible(state, k):
                        mask[self.encode_action(DEPOT, v_idx)] = True

            else:  # DRONE
                if state.drone_active[k]:
                    # airborne: can extend trip or land
                    for j in range(1, N1):
                        if self._drone_extend_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True
                    # landing nodes: depot and truck's current node
                    for land in self._landing_nodes(state, k):
                        if self._drone_land_feasible(state, k, land):
                            mask[self.encode_action(land, v_idx)] = True
                else:
                    # idle: can start a new trip to any feasible customer
                    for j in range(1, N1):
                        if self._drone_launch_feasible(state, k, j):
                            mask[self.encode_action(j, v_idx)] = True

        return ActionMask.from_bool_array(mask)

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    def _landing_nodes(self, state: VRPBTWState, k: int) -> List[int]:
        """Valid landing nodes: depot + truck's current node (if not depot)."""
        nodes = [DEPOT]
        t_node = int(state.truck_node[k])
        if t_node != DEPOT:
            nodes.append(t_node)
        return nodes

    def _phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check demand sign matches truck phase for fleet k."""
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
        if abs(self.demands[j]) > state.truck_load[k]:
            return False
        from_node = int(state.truck_node[k])
        dist = self.dist_matrix[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        if arrive > self.tw_close[j]:
            return False
        depart = max(arrive, self.tw_open[j]) + self.service_times[j]
        # must be able to return to depot after service
        if depart + self.dist_matrix[j, DEPOT] / self.v_t > self.T_max:
            return False
        return True

    def _truck_return_feasible(self, state: VRPBTWState, k: int) -> bool:
        from_node = int(state.truck_node[k])
        if from_node == DEPOT:
            return False
        arrive = state.truck_time[k] + self.dist_matrix[from_node, DEPOT] / self.v_t
        return arrive <= self.T_max

    def _nearest_return_dist(self, state: VRPBTWState, k: int, from_node: int) -> float:
        """Min dist from from_node to any valid landing node."""
        candidates = [self.dist_matrix[from_node, DEPOT]]
        t_node = int(state.truck_node[k])
        if t_node != DEPOT:
            candidates.append(self.dist_matrix[from_node, t_node])
        return float(min(candidates))

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Drone is idle. Can it start a trip serving customer j?

        Drone may launch from its current node provided that node is a
        valid launch point: depot (always valid) or the truck's current
        node (co-location required for non-depot launch).
        """
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        if abs(self.demands[j]) > state.drone_load[k]:
            return False
        # valid launch points: depot OR truck's current node
        drone_at = int(state.drone_node[k])
        valid_launch = drone_at == DEPOT or drone_at == int(state.truck_node[k])
        if not valid_launch:
            return False
        from_node = int(state.drone_node[k])
        dist_out = self.dist_matrix[from_node, j]
        # minimum trip: out to j + back to nearest landing
        dist_back = self._nearest_return_dist(state, k, j)
        trip_total = dist_out + dist_back
        if trip_total > self.D_max:
            return False
        # time window
        depart_t = state.drone_time[k] + self.launch_time
        arrive_j = depart_t + dist_out / self.v_d
        if arrive_j > self.tw_close[j]:
            return False
        return True

    def _drone_extend_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Drone is airborne. Can it fly to customer j next?"""
        if not state.drone_active[k] or state.served[j]:
            return False
        if not self._phase_ok(state, k, j):
            return False
        if abs(self.demands[j]) > state.drone_load[k]:
            return False
        from_node = int(state.drone_node[k])
        dist_to_j = self.dist_matrix[from_node, j]
        dist_back = self._nearest_return_dist(state, k, j)
        new_total = state.drone_dist[k] + dist_to_j + dist_back
        if new_total > self.D_max:
            return False
        # time window
        arrive_j = state.drone_time[k] + dist_to_j / self.v_d
        if arrive_j > self.tw_close[j]:
            return False
        return True

    def _drone_land_feasible(self, state: VRPBTWState, k: int, land: int) -> bool:
        """Drone is airborne. Can it land at node land?"""
        if not state.drone_active[k]:
            return False
        from_node = int(state.drone_node[k])
        dist_back = self.dist_matrix[from_node, land]
        new_total = state.drone_dist[k] + dist_back
        if new_total > self.D_max:
            return False
        arrive = state.drone_time[k] + dist_back / self.v_d + self.land_time
        if arrive > self.T_max:
            return False
        return True

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        node, v_idx = self.decode_action(action)
        k, vtype = self.vehicle_fleet(v_idx)
        state = _copy_state(state)

        if vtype == TRUCK:
            reward = self._apply_truck(state, k, node)
        else:
            if state.drone_active[k]:
                # airborne: extend trip or land
                if node != DEPOT and node != int(state.truck_node[k]):
                    reward = self._apply_drone_extend(state, k, node)
                else:
                    reward = self._apply_drone_land(state, k, node)
            else:
                # idle: launch new trip
                reward = self._apply_drone_launch(state, k, node)

        terminated = self._check_terminated(state)
        if terminated:
            reward += self._terminal_reward(state)

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.action_space_size)
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
                "node": node,
                "fleet": k,
                "vehicle": "truck" if vtype == TRUCK else "drone",
                "served_count": int(state.served[1:].sum()),
            },
        )

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> float:
        from_node = int(state.truck_node[k])
        dist = self.dist_matrix[from_node, j]
        arrive = state.truck_time[k] + dist / self.v_t
        serve_start = max(arrive, self.tw_open[j]) if j != DEPOT else arrive
        tardiness = max(serve_start - self.tw_close[j], 0.0) if j != DEPOT else 0.0

        state.truck_time[k] = serve_start + self.service_times[j]
        state.truck_node[k] = j
        state.truck_load[k] -= abs(self.demands[j])
        state.truck_routes[k].append(j)

        if j != DEPOT:
            state.served[j] = True
            self._update_truck_phase(state, k)

        # add truck backbone edge to graph
        self._add_edge(
            state,
            from_node,
            j,
            k,
            TRUCK,
            depart_time=state.truck_time[k]
            - self.service_times[j]
            - (serve_start - arrive),
            arrive_time=arrive,
            tardiness=tardiness,
        )

        return float(-(self.c_t * dist) - self.lambda_weight * tardiness)

    def _apply_drone_launch(self, state: VRPBTWState, k: int, j: int) -> float:
        from_node = int(state.drone_node[k])
        dist_out = self.dist_matrix[from_node, j]
        depart_t = state.drone_time[k] + self.launch_time
        arrive_j = depart_t + dist_out / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)

        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= abs(self.demands[j])
        state.drone_dist[k] += dist_out
        state.drone_active[k] = True
        state.served[j] = True
        self._update_truck_phase(state, k)

        # route: append launch node then first customer
        launch_node = int(state.truck_node[k])
        state.drone_route_nodes[k].append(launch_node)
        state.drone_route_mask[k].append(0)
        state.drone_route_nodes[k].append(j)
        state.drone_route_mask[k].append(1)

        self._add_edge(
            state,
            launch_node,
            j,
            k,
            DRONE,
            depart_time=depart_t,
            arrive_time=arrive_j,
            tardiness=tardiness,
        )

        return float(-(self.c_d * dist_out) - self.lambda_weight * tardiness)

    def _apply_drone_extend(self, state: VRPBTWState, k: int, j: int) -> float:
        from_node = int(state.drone_node[k])
        dist_to_j = self.dist_matrix[from_node, j]
        arrive_j = state.drone_time[k] + dist_to_j / self.v_d
        serve_start = max(arrive_j, self.tw_open[j])
        tardiness = max(arrive_j - self.tw_close[j], 0.0)

        state.drone_time[k] = serve_start + self.service_times[j]
        state.drone_node[k] = j
        state.drone_load[k] -= abs(self.demands[j])
        state.drone_dist[k] += dist_to_j
        state.served[j] = True
        self._update_truck_phase(state, k)

        state.drone_route_nodes[k].append(j)
        state.drone_route_mask[k].append(1)

        self._add_edge(
            state,
            from_node,
            j,
            k,
            DRONE,
            depart_time=state.drone_time[k] - self.service_times[j],
            arrive_time=arrive_j,
            tardiness=tardiness,
        )

        return float(-(self.c_d * dist_to_j) - self.lambda_weight * tardiness)

    def _apply_drone_land(self, state: VRPBTWState, k: int, land: int) -> float:
        from_node = int(state.drone_node[k])
        dist_back = self.dist_matrix[from_node, land]
        arrive = state.drone_time[k] + dist_back / self.v_d + self.land_time

        state.drone_time[k] = max(
            arrive, state.truck_time[k] if land == state.truck_node[k] else arrive
        )
        state.drone_node[k] = land
        state.drone_dist[k] += dist_back
        state.drone_active[k] = False
        # reset per-trip budget
        state.drone_load[k] = self.Q_d
        state.drone_dist[k] = 0.0

        state.drone_route_nodes[k].append(land)
        state.drone_route_mask[k].append(0)

        self._add_edge(
            state,
            from_node,
            land,
            k,
            DRONE,
            depart_time=state.drone_time[k] - dist_back / self.v_d - self.land_time,
            arrive_time=arrive,
            tardiness=0.0,
        )

        return float(-(self.c_d * dist_back))

    # ------------------------------------------------------------------
    # Graph edge construction
    # ------------------------------------------------------------------

    def _add_edge(
        self,
        state: VRPBTWState,
        src: int,
        dst: int,
        fleet: int,
        vtype: int,
        depart_time: float,
        arrive_time: float,
        tardiness: float,
    ) -> None:
        dist = self.dist_matrix[src, dst]
        travel_time = dist / (self.v_t if vtype == TRUCK else self.v_d)
        feat = np.array(
            [
                float(vtype),  # 0: edge_type
                travel_time / (self.T_max + 1e-6),  # 1: travel_time norm
                dist / (self.D_max + 1e-6),  # 2: travel_dist norm
                depart_time / (self.T_max + 1e-6),  # 3: depart_time norm
                arrive_time / (self.T_max + 1e-6),  # 4: arrive_time norm
                tardiness / (self.T_max + 1e-6),  # 5: tardiness norm
            ],
            dtype=np.float32,
        )

        new_edge = np.array([[src], [dst]], dtype=np.int32)
        state.edge_index = np.concatenate([state.edge_index, new_edge], axis=1)
        state.edge_attr = np.concatenate([state.edge_attr, feat[None]], axis=0)
        state.edge_fleet = np.concatenate([state.edge_fleet, [fleet]], axis=0)

    # ------------------------------------------------------------------
    # Phase update
    # ------------------------------------------------------------------

    def _update_truck_phase(self, state: VRPBTWState, k: int) -> None:
        """Flip truck k to backhaul phase when all linehaul customers served."""
        if state.truck_phase[k] == 0 and len(self._linehaul_idx) > 0:
            if state.served[self._linehaul_idx].all():
                state.truck_phase[k] = 1

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_terminated(self, state: VRPBTWState) -> bool:
        if not state.served[1:].all():
            return False
        if state.drone_active.any():
            return False
        return True

    def _terminal_reward(self, state: VRPBTWState) -> float:
        unserved = int((~state.served[1:]).sum())
        return float(10.0 if unserved == 0 else -2.0 * unserved)

    # ------------------------------------------------------------------
    # state_to_obs
    # ------------------------------------------------------------------

    def state_to_obs(self, state: VRPBTWState) -> Dict[str, np.ndarray]:
        N1 = self.n_customers + 1
        T = self.T_max + 1e-6
        max_coord = float(self.coords.max()) + 1e-6

        node_feat = np.stack(
            [
                self.coords[:, 0] / max_coord,
                self.coords[:, 1] / max_coord,
                self.demands / (self.Q_t + 1e-6),
                self.tw_open / T,
                self.tw_close / T,
                np.clip((self.tw_close - self.tw_open) / T, 0.0, 1.0),
                self.service_times / T,
                state.served.astype(np.float32),
                np.full(N1, self.lambda_weight, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)  # (N+1, NODE_FEAT_DIM)

        veh_rows = []
        for k in range(self.K):
            # truck row
            veh_rows.append(
                np.array(
                    [
                        state.truck_node[k] / max(N1 - 1, 1),
                        state.truck_load[k] / (self.Q_t + 1e-6),
                        state.truck_time[k] / T,
                        float(state.truck_phase[k]),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                )
            )
            # drone row
            veh_rows.append(
                np.array(
                    [
                        state.drone_node[k] / max(N1 - 1, 1),
                        state.drone_load[k] / (self.Q_d + 1e-6),
                        state.drone_time[k] / T,
                        0.0,
                        (self.D_max - state.drone_dist[k]) / (self.D_max + 1e-6),
                        float(state.drone_active[k]),
                    ],
                    dtype=np.float32,
                )
            )

        vehicle_feat = np.stack(veh_rows, axis=0)  # (2K, VEH_FEAT_DIM)

        return {
            "node_features": node_feat,
            "vehicle_features": vehicle_feat,
            "edge_index": state.edge_index.copy(),
            "edge_attr": state.edge_attr.copy(),
            "edge_fleet": state.edge_fleet.copy(),
        }

    # ------------------------------------------------------------------
    # evaluate / is_complete / properties
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> float:
        # recompute from routes to avoid floating point accumulation
        total_cost = 0.0
        for k in range(self.K):
            prev = DEPOT
            for j in state.truck_routes[k]:
                total_cost += self.c_t * self.dist_matrix[prev, j]
                prev = j
            prev = DEPOT
            for idx, j in enumerate(state.drone_route_nodes[k]):
                total_cost += self.c_d * self.dist_matrix[prev, j]
                prev = j
        unserved = int((~state.served[1:]).sum())
        return -(total_cost + unserved * 1000.0)

    def is_complete(self, state: VRPBTWState) -> bool:
        return self._check_terminated(state)

    @property
    def action_space_size(self) -> int:
        return (self.n_customers + 1) * 2 * self.K

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # primary obs shape for buffer allocation (node features)
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

    # ------------------------------------------------------------------
    # decode_solution
    # ------------------------------------------------------------------

    def decode_solution(self, state: VRPBTWState) -> Solution:
        return Solution(
            problem_name=self.name,
            raw_state=state,
            objective=self.evaluate(state),
            metadata={
                "served_count": int(state.served[1:].sum()),
                "n_customers": self.n_customers,
                "truck_routes": [list(r) for r in state.truck_routes],
                "drone_route_nodes": [list(r) for r in state.drone_route_nodes],
                "drone_route_mask": [list(m) for m in state.drone_route_mask],
                "unserved": int((~state.served[1:]).sum()),
            },
        )

    def heuristic_solution(self) -> Optional[float]:
        """Simple nearest-neighbour truck-only heuristic as baseline."""
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        current, time, dist, load = DEPOT, 0.0, 0.0, self.Q_t
        phase = 0
        while not served[1:].all():
            best_j, best_d = -1, float("inf")
            for j in range(1, self.n_customers + 1):
                if served[j]:
                    continue
                if phase == 0 and self.demands[j] < 0:
                    continue
                if phase == 1 and self.demands[j] > 0:
                    continue
                if abs(self.demands[j]) > load:
                    continue
                d = self.dist_matrix[current, j]
                arrive = time + d / self.v_t
                if arrive > self.tw_close[j]:
                    continue
                if d < best_d:
                    best_d, best_j = d, j
            if best_j == -1:
                if phase == 0:
                    phase = 1
                    continue
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
            if phase == 0 and len(self._linehaul_idx) > 0:
                if served[self._linehaul_idx].all():
                    phase = 1
        dist += self.dist_matrix[current, DEPOT]
        unserved = int((~served[1:]).sum())
        return -(dist * self.c_t + unserved * 1000.0)


# ---------------------------------------------------------------------------
# State copy
# ---------------------------------------------------------------------------


def _copy_state(s: VRPBTWState) -> VRPBTWState:
    return VRPBTWState(
        truck_node=s.truck_node.copy(),
        truck_time=s.truck_time.copy(),
        truck_load=s.truck_load.copy(),
        truck_phase=s.truck_phase.copy(),
        drone_node=s.drone_node.copy(),
        drone_time=s.drone_time.copy(),
        drone_load=s.drone_load.copy(),
        drone_dist=s.drone_dist.copy(),
        drone_active=s.drone_active.copy(),
        served=s.served.copy(),
        truck_routes=[list(r) for r in s.truck_routes],
        drone_route_nodes=[list(r) for r in s.drone_route_nodes],
        drone_route_mask=[list(m) for m in s.drone_route_mask],
        edge_index=s.edge_index.copy(),
        edge_attr=s.edge_attr.copy(),
        edge_fleet=s.edge_fleet.copy(),
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
            rng.uniform(1.0, 10.0, n_linehaul),  # positive: linehaul
            -rng.uniform(1.0, 10.0, n_backhaul),  # negative: backhaul
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
    # D_max is a distance budget (divided by drone speed to get time).
    # Set to grid diagonal * 1.2 so multi-stop trips are feasible.
    drone_trip_duration = grid_size * np.sqrt(2) * 1.2

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
