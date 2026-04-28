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
    drone_phase: (
        np.ndarray
    )  # (K,) phase when drone k launched (locked for trip duration)

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
        drone_phase=s.drone_phase.copy(),
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
        # Task ID (set when reset() is called with task_id)
        self.task_id: Optional[str] = None

        # Available tasks from config (task_id format: "{difficulty}_N{customers}_F{fleets}_{distribution}")
        self.tasks: List[str] = cfg.get("tasks", [])

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
        self.service_time: float = float(cfg.get("service_time_min", 5.0)) / 60.0

        # Instance generation parameters (for _generate_instance)
        self.demand_range_linehaul: Tuple[int, int] = (
            int(cfg.get("demand_range_linehaul_min", 5)),
            int(cfg.get("demand_range_linehaul_max", 10)),
        )
        self.demand_range_backhaul: Tuple[int, int] = (
            int(cfg.get("demand_range_backhaul_min", 5)),
            int(cfg.get("demand_range_backhaul_max", 10)),
        )
        self.time_window_scaling_factor: float = float(
            cfg.get("time_window_scaling_factor", 1.0)
        )

        self._linehaul_idx: np.ndarray = np.array([], dtype=np.int32)
        self._backhaul_idx: np.ndarray = np.array([], dtype=np.int32)

    @classmethod
    def from_config(cls, cfg: Dict) -> "VRPBTWEnv":
        """Factory method: instantiate VRPBTWEnv from config dict.

        Instance-specific parameters (n_customers, n_fleets) are set dynamically
        in reset() via _generate_instance() when task_id is provided.
        """
        # Support both old (flat) and new (properties) structure
        props = cfg.get("properties", cfg)
        return cls(props)

    # ------------------------------------------------------------------
    # Instance generation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_task_id(task_id: str) -> Tuple[int, int, str]:
        """Parse task ID format: {difficulty}_N{customers}_F{fleets}_{distribution}.

        Returns: (n_customers, n_fleets, distribution)
        Example: "001_N10_F2_RC" -> (10, 2, "RC")
        """
        parts = task_id.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid task_id format: {task_id!r}")

        try:
            n_customers = int(parts[1][1:])  # Skip 'N'
            n_fleets = int(parts[2][1:])  # Skip 'F'
            dist = parts[3]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse task_id {task_id!r}: {e}")

        return n_customers, n_fleets, dist

    def _generate_coords(self, num_customers: int, dist_type: str) -> List[List[float]]:
        """Generate customer coordinates based on distribution type (R, C, RC).
        Uses global numpy RNG state set by seed_everything().
        """
        if dist_type == "R":
            return np.random.uniform(
                0, self.max_coord, size=(num_customers, 2)
            ).tolist()

        elif dist_type == "C":
            coords = []
            remaining_nodes = num_customers

            if num_customers <= 200:
                std_dev = self.max_coord / 25
            elif num_customers <= 400:
                std_dev = self.max_coord / 32
            else:
                std_dev = self.max_coord / 40
            min_dist = std_dev * 4

            centers = []
            while remaining_nodes > 0:
                current_cluster_size = min(
                    remaining_nodes, int(np.random.uniform(10, 16))
                )

                proposal = np.random.uniform(5, self.max_coord - 5, size=(2,))
                valid_center = False
                attempts = 0
                while not valid_center and attempts < 1000:
                    proposal = np.random.uniform(5, self.max_coord - 5, size=(2,))
                    if not centers:
                        valid_center = True
                    else:
                        dists = [
                            np.linalg.norm(proposal - np.array(c)) for c in centers
                        ]
                        if min(dists) >= min_dist:
                            valid_center = True
                    attempts += 1

                centers.append(proposal.tolist())

                for _ in range(current_cluster_size):
                    point = np.random.normal(proposal, std_dev)
                    coords.append(np.clip(point, 0, self.max_coord).tolist())

                remaining_nodes -= current_cluster_size

            return coords

        elif dist_type == "RC":
            n_c = num_customers // 2
            return self._generate_coords(n_c, "C") + self._generate_coords(
                num_customers - n_c, "R"
            )

        raise ValueError(f"Unknown distribution type: {dist_type}")

    def _generate_instance(self, task_id: str) -> Dict[str, Any]:
        """Generate modifiable VRPBTW instance attributes from task_id.

        Args:
            task_id: Task ID in format "{difficulty}_N{customers}_F{fleets}_{distribution}"

        Returns:
            Dict with modifiable attributes (depot, customers, n_fleets).
            Static attributes (capacities, speeds, etc.) are already on self.

        Uses global numpy RNG state set by seed_everything().
        """
        n_customers, n_fleets, dist_type = self._parse_task_id(task_id)

        # Generate coordinates using global numpy RNG (seeded by seed_everything())
        coords = self._generate_coords(n_customers, dist_type)
        depot_coord = [self.max_coord / 2.0, self.max_coord / 2.0]

        # Generate customers with time windows and demands
        customers = []
        linehaul_count = int(n_customers * 0.5)
        types = ["LINEHAUL"] * linehaul_count + ["BACKHAUL"] * (
            n_customers - linehaul_count
        )
        np.random.shuffle(types)

        for i in range(n_customers):
            node_type = types[i]
            demand_range = (
                self.demand_range_linehaul
                if node_type == "LINEHAUL"
                else self.demand_range_backhaul
            )

            dist_km = np.linalg.norm(np.array(coords[i]) - np.array(depot_coord))
            min_reach_time = dist_km / self.v_t
            ready_h = float(np.random.uniform(min_reach_time * 1.1, self.T_max * 0.7))
            width_h = float(
                np.random.uniform(
                    1.0,
                    1.0
                    + (self.time_window_scaling_factor * (dist_km / self.max_coord)),
                )
            )

            demand = float(int(np.random.randint(demand_range[0], demand_range[1] + 1)))
            if node_type == "BACKHAUL":
                demand = -demand
            customers.append(
                [
                    float(coords[i][0]),
                    float(coords[i][1]),
                    float(round(ready_h, 4)),
                    float(round(min(ready_h + width_h, self.T_max), 4)),
                    demand,
                ]
            )

        return {
            "depot": depot_coord,
            "customers": customers,
            "n_fleets": n_fleets,
        }

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
        tw_close_all = np.concatenate([[self.T_max], customers[:, 3]])
        demands_all = np.concatenate([[0.0], customers[:, 4]])
        svc_all = np.full(self.n_customers + 1, self.service_time, dtype=np.float32)
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

        cust_d = self.demands[1:]
        self._linehaul_idx = (np.where(cust_d > 0)[0] + 1).astype(np.int32)
        self._backhaul_idx = (np.where(cust_d < 0)[0] + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # reset override — computes nadir internally before returning state
    # ------------------------------------------------------------------

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

    def compute_return(self) -> float:
        """Compute return from current solution: -objective / (max_dist * max_cost_unit).

        Gets solution from current state, extracts objective, negates it to convert
        from cost minimization to return maximization, and normalizes by both spatial
        and cost scales for scale-independence across instances.

        Returns
        -------
        float
            Normalized return in range [-(N^2), ~0]:
            - Worst (no customers served): ~-(N^2) where N = n_customers
            - Best (all served, minimal cost): ~0 or slightly negative
        """
        solution = self.current_solution()
        normalized_factor = (
            2.0 * self.max_coord * max(self.c_t, self.c_d) * self.n_customers
        )
        return -solution.objective / normalized_factor

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
            drone_phase=np.zeros(K, dtype=np.int32),
            served=served,
            current_cost=0.0,
            current_max_tard=0.0,
            truck_routes=[[DEPOT] for _ in range(K)],
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
                    for land_idx in self._landing_nodes(state, k):
                        if self._drone_land_feasible(state, k, land_idx):
                            land_node = state.truck_routes[k][land_idx]
                            mask[self.encode_action(land_node, v_idx)] = True
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
        Return indices into truck_routes[k] where drone can land.
        Drone can land at nodes visited strictly after launch node.
        Node at index i retrieved via: truck_routes[k][i]
        """
        launch_node = int(state.drone_launch_node[k])
        route = state.truck_routes[k]

        # Find the last position of the launch node in the truck route
        launch_idx = -1
        for i in range(len(route) - 1, -1, -1):
            if route[i] == launch_node:
                launch_idx = i
                break

        # Return indices of all nodes after launch
        return list(range(launch_idx + 1, len(route)))

    def _late_land_feasible(
        self, state: VRPBTWState, k: int, land_idx: int, drone_land_time: float
    ) -> bool:
        """
        If drone lands at truck_routes[k][land_idx] at drone_land_time (possibly late),
        check if truck can still service all subsequent nodes without missing customers.

        Simulates truck's remaining route from landing node with new ready time.
        """
        route = state.truck_routes[k]
        land_node = route[land_idx]
        t = drone_land_time
        node = land_node

        # Simulate remaining route from landing node onwards
        for i in range(land_idx + 1, len(route)):
            next_node = route[i]
            dist = self.manhattan_dist[node, next_node]
            arrive = t + dist / self.v_t

            # Check time window feasibility for next_node
            if arrive > self.tw_close[next_node]:
                return False  # Can't reach next_node on time

            service_start = max(arrive, self.tw_open[next_node])
            t = service_start + self.service_times[next_node]
            node = next_node

        # Check return to depot is feasible
        dist_to_depot = self.manhattan_dist[node, DEPOT]
        return t + dist_to_depot / self.v_t <= self.T_max

    def _truck_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check if truck can serve node j based on truck's current phase."""
        phase = int(state.truck_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _drone_phase_ok(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check if drone can serve node j based on drone's locked launch phase."""
        phase = int(state.drone_phase[k])
        demand = self.demands[j]
        if phase == 0 and demand < 0:
            return False
        if phase == 1 and demand > 0:
            return False
        return True

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if j == DEPOT or state.served[j]:
            return False
        # Prevent serving after returning to depot: if truck is at DEPOT and route has more than initial DEPOT,
        # the truck has already returned and cannot serve more
        if int(state.truck_node[k]) == DEPOT and len(state.truck_routes[k]) > 1:
            return False
        if not self._truck_phase_ok(state, k, j):
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
        """Minimum flight time from from_node to any unserved node.

        Returns the minimum time for drone to reach a landing node (where truck can be).
        If no feasible landing node exists, returns infinity.
        """
        unserved_nodes = ~state.served
        reachable_nodes = np.where(unserved_nodes)[0]
        if len(reachable_nodes) == 0:
            return float("inf")
        distances = self.euclidean_dist[from_node, reachable_nodes]
        return float(np.min(distances)) / self.v_d

    def _drone_launch_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        if state.drone_active[k] or state.served[j]:
            return False
        if not self._truck_phase_ok(state, k, j):
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
        if not self._drone_phase_ok(state, k, j):
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

    def _drone_land_feasible(self, state: VRPBTWState, k: int, land_idx: int) -> bool:
        """Check if drone can land at truck_routes[k][land_idx] without losing customers.

        Args:
            land_idx: index into truck_routes[k]
        """
        if not state.drone_active[k]:
            return False

        land_node = state.truck_routes[k][land_idx]
        from_node = int(state.drone_node[k])
        t_back = self.euclidean_dist[from_node, land_node] / self.v_d
        elapsed = self._elapsed_trip_time(state, k)

        if elapsed + t_back > self.t_max:
            return False

        drone_land_time = state.drone_time[k] + t_back + self.land_time
        if drone_land_time > self.T_max:
            return False

        # Check if late landing would cause truck to miss future customers
        return self._late_land_feasible(state, k, land_idx, drone_land_time)

    # ------------------------------------------------------------------
    # Infeasible action handling
    # ------------------------------------------------------------------

    def _compute_drone_trip_cost(self, state: VRPBTWState, k: int) -> float:
        """Compute total euclidean distance cost of current drone trip."""
        cost = 0.0
        prev = DEPOT
        for node in state.drone_route_nodes[k]:
            cost += self.euclidean_dist[prev, node]
            prev = node
        return cost

    def _rollback_drone_trip(self, state: VRPBTWState, k: int) -> float:
        """
        Rollback entire drone trip: undo served nodes, reset drone to inactive.

        Returns the distance cost incurred by the trip (for reward calculation).
        """
        trip_cost = self._compute_drone_trip_cost(state, k)

        # Unserve all nodes visited on this trip
        for node in state.drone_route_nodes[k]:
            if node != DEPOT and 1 <= node < len(state.served):
                state.served[node] = False

        # Reset drone to inactive state at truck location with truck's current time
        state.drone_active[k] = False
        state.drone_node[k] = state.truck_node[k]
        state.drone_load[k] = self.Q_d
        state.drone_time[k] = state.truck_time[k]

        # Clear trip routes
        state.drone_route_nodes[k] = []
        state.drone_route_mask[k] = []

        # Revert trip cost
        state.current_cost -= trip_cost

        return trip_cost

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        node, v_idx = self.decode_action(action)
        k, vtype = self.vehicle_fleet_type(v_idx)

        state = _copy_state(state)

        # Apply action (assumed feasible — policy ensures this)
        if vtype == TRUCK:
            self._apply_truck(state, k, node)
        else:  # DRONE
            if state.drone_active[k]:
                landing_indices = self._landing_nodes(state, k)
                # Check if node is a valid landing location
                is_landing = any(
                    state.truck_routes[k][idx] == node for idx in landing_indices
                )

                if is_landing:
                    self._apply_drone_land(state, k, node)
                else:
                    self._apply_drone_extend(state, k, node)
            else:
                self._apply_drone_launch(state, k, node)

        terminated = self._is_terminated(state)

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.action_space_size)
        )
        if not terminated and next_mask.is_empty():
            # Remove incomplete drone trips before terminating
            for k in range(self.K):
                if state.drone_active[k]:
                    launch_node = int(state.drone_launch_node[k])
                    route = state.drone_route_nodes[k]
                    mask = state.drone_route_mask[k]

                    # Find where this trip started (last occurrence of launch_node marked as 0)
                    launch_idx = -1
                    for i in range(len(route) - 1, -1, -1):
                        if route[i] == launch_node and mask[i] == 0:
                            launch_idx = i
                            break

                    if launch_idx >= 0:
                        # Unserve customers served in incomplete trip
                        for i in range(launch_idx + 1, len(route)):
                            if mask[i] == 1:
                                node = route[i]
                                state.served[node] = False

                        # Remove incomplete trip from route
                        state.drone_route_nodes[k] = state.drone_route_nodes[k][
                            : launch_idx + 1
                        ]
                        state.drone_route_mask[k] = state.drone_route_mask[k][
                            : launch_idx + 1
                        ]

                    # Reset drone to idle at launch node, load based on phase
                    phase = int(state.truck_phase[k])
                    state.drone_active[k] = False
                    state.drone_phase[k] = phase
                    state.drone_node[k] = launch_node
                    state.drone_load[k] = self.Q_d if phase == 0 else 0.0

            terminated = True

        return StepResult(
            next_state=state,
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
        state.drone_phase[k] = int(state.truck_phase[k])
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
        state.drone_phase[k] = int(state.truck_phase[k])
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
        drones_idle = not state.drone_active.any()
        trucks_at_depot = bool((state.truck_node == DEPOT).all())
        return drones_idle and trucks_at_depot

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
        T = max_dist / min(
            self.v_t, self.v_d
        )  # time to traverse longest path at slowest speed

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

        keep_truck = ~state.served  # unserved customers
        keep_truck[DEPOT] = True
        for k in range(self.K):
            keep_truck[int(state.truck_node[k])] = True  # current truck pos
            if state.drone_active[k]:
                for lc_idx in self._landing_nodes(
                    state, k
                ):  # landing candidates (route indices)
                    lc_node = state.truck_routes[k][lc_idx]
                    keep_truck[lc_node] = True

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

    def get_starting_actions(self) -> np.ndarray:
        """
        Return candidate first actions for POMO: linehaul customers with first fleet (truck and drone).

        For backhaul VRP, linehaul customers must be served first, and only the first
        fleet can make the initial dispatch. This includes both truck and drone options
        for each linehaul customer, reducing the candidate action space.

        Vehicle indices for first fleet (fleet 0):
        - Truck: vehicle_idx = 0
        - Drone: vehicle_idx = K

        Returns
        -------
        np.ndarray: Indices of candidate actions (linehaul customers with first fleet vehicles).
        """
        starting_actions = []
        for j in self._linehaul_idx:
            # Truck for first fleet (0)
            starting_actions.append(self.encode_action(int(j), 0))
            # Drone for first fleet (K)
            starting_actions.append(self.encode_action(int(j), self.K))
        return np.array(starting_actions, dtype=np.int64)
