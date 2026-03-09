"""
examples/vrpbtw_problem.py
--------------------------
Vehicle Routing Problem with Back-hauling and Time Windows (VRPBTW)
solved with a heterogeneous truck-drone fleet.

Problem setup
-------------
  - One depot (node 0)
  - N customers, each with: 2D coordinate, time window [open, close], demand
  - K fleets, each fleet = one truck + one drone
  - Truck and drone have separate capacities, speeds, and cost rates
  - Drone sortie overhead is split into launch_time (lift-off) and land_time
    (rendezvous + securing) — these are intentionally kept separate
  - System duration T_max: global wall-clock limit for all vehicles
  - Drone trip duration D_max: max airborne time per sortie (battery limit)
  - Service time s: time spent at each customer node (both vehicles)

Action formulation  (sequential)
---------------------------------
At every decision step, ONE (fleet, vehicle) pair is chosen to serve the
NEXT unserved customer.  The agent picks a flat integer:

    action ∈ [0, K*2*(N+1) )

which decodes as:
    fleet_id    = action // (2 * (N+1))
    vehicle_id  = (action // (N+1)) % 2   (0=truck, 1=drone)
    customer_id = action % (N+1)           (0=depot/return, 1..N=customers)

Drone synchronisation
---------------------
When a drone is dispatched to customer j from the truck's current position p:
  1. Truck continues along its route while drone is airborne.
  2. Drone flies  p → j → rendezvous_node r.
  3. Rendezvous node r must be the truck's NEXT scheduled stop (or depot).
  4. Drone arrival at r  = drone_depart + dist(p,j)/v_d + launch_time
                           + service_time + dist(j,r)/v_d + land_time
  5. Truck must wait at r if it arrives before the drone.
  The waiting cost is counted as idle truck time (no extra distance cost).

Reward
------
Dense: −(truck_cost * truck_dist_step + drone_cost * drone_dist_step)
       + time_window_bonus (small positive if arrived inside window)
Terminal: large bonus if all customers served, penalty per unserved customer.

Observation  shape = (N+1, NODE_FEAT_DIM)
--------------------------------------
Node features (depot + N customers):
  [x, y, tw_open/T, tw_close/T, demand/Q_t, served,
   dist_nearest_truck, dist_nearest_drone, time_slack/T,
   is_depot]                          → NODE_FEAT_DIM = 10
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.problem import CombinatorialProblem, ActionMask, StepResult
from core.solution import Solution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 10  # dimension of per-node observation vector
TRUCK = 0
DRONE = 1
DEPOT = 0  # node index of the depot


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class VRPBTWState:
    # ---- Truck state  (one per fleet) ----------------------------------
    truck_pos: np.ndarray  # (K, 2)  current 2D coordinates
    truck_load: np.ndarray  # (K,)    remaining capacity
    truck_time: np.ndarray  # (K,)    current clock time
    truck_dist: np.ndarray  # (K,)    total distance travelled
    truck_route: List[List[int]]  # (K,) visited node sequence

    # ---- Drone state  (one per fleet) ----------------------------------
    drone_pos: np.ndarray  # (K, 2)  current 2D coordinates
    drone_load: np.ndarray  # (K,)    remaining capacity
    drone_time: np.ndarray  # (K,)    earliest time drone is available
    drone_dist: np.ndarray  # (K,)    total distance flown
    drone_airtime: np.ndarray  # (K,)    cumulative airborne time in current sortie
    drone_active: np.ndarray  # (K,)    bool — currently in flight?
    drone_rendezvous: (
        np.ndarray
    )  # (K,)    int node index where drone will land (-1 if idle)
    drone_route: List[List[int]]  # (K,)

    # ---- Global --------------------------------------------------------
    served: np.ndarray  # (N+1,)  bool — node served? (index 0=depot, always True)
    step: int


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class VRPBTWProblem(CombinatorialProblem):
    """
    VRPBTW with truck-drone fleets.

    Instance dict keys
    ------------------
    depot          : [x, y]
    customers      : [[x, y, tw_open, tw_close, demand], ...]   length N
    n_fleets       : int
    truck_capacity : float
    drone_capacity : float
    system_duration: float   (T_max)
    drone_trip_duration : float  (D_max — battery / max airborne time)
    truck_speed    : float
    drone_speed    : float
    truck_cost     : float   (per distance unit)
    drone_cost     : float   (per distance unit)
    launch_time    : float   (time to lift-off drone from truck)
    land_time      : float   (time to land/secure drone back on truck)
    service_time   : float   (dwell time at each customer, same for both vehicles)
    """

    def __init__(self, n_customers: int = 10, n_fleets: int = 2):
        super().__init__(name="VRPBTW")
        self.n_customers = n_customers
        self.n_fleets = n_fleets

        # Populated by encode_instance
        self.coords: np.ndarray = np.array([])  # (N+1, 2)  depot + customers
        self.tw_open: np.ndarray = np.array([])  # (N+1,)
        self.tw_close: np.ndarray = np.array([])  # (N+1,)
        self.demands: np.ndarray = np.array([])  # (N+1,)
        self.dist_matrix: np.ndarray = np.array([])  # (N+1, N+1)

        # Scalar parameters
        self.K: int = n_fleets
        self.Q_t: float = 0.0
        self.Q_d: float = 0.0
        self.T_max: float = 0.0
        self.D_max: float = 0.0
        self.v_t: float = 0.0
        self.v_d: float = 0.0
        self.c_t: float = 0.0
        self.c_d: float = 0.0
        self.launch_time: float = 0.0  # ← separate from land_time
        self.land_time: float = 0.0  # ← separate from launch_time
        self.service_time: float = 0.0

    # ------------------------------------------------------------------
    # 1. encode_instance
    # ------------------------------------------------------------------

    def encode_instance(self, raw_instance: Dict) -> None:
        depot = np.array(raw_instance["depot"], dtype=np.float32)
        customers = np.array(raw_instance["customers"], dtype=np.float32)  # (N, 5)

        self.n_customers = len(customers)
        self.K = int(raw_instance["n_fleets"])

        # Node arrays: index 0 = depot, 1..N = customers
        coords_all = np.vstack([depot, customers[:, :2]])  # (N+1, 2)
        tw_open_all = np.concatenate([[0.0], customers[:, 2]])  # depot open=0
        tw_close_all = np.concatenate(
            [[raw_instance["system_duration"]], customers[:, 3]]
        )  # depot close=T_max
        demands_all = np.concatenate([[0.0], customers[:, 4]])  # depot demand=0

        self.coords = coords_all.astype(np.float32)
        self.tw_open = tw_open_all.astype(np.float32)
        self.tw_close = tw_close_all.astype(np.float32)
        self.demands = demands_all.astype(np.float32)

        # Distance matrix  (Euclidean)
        diff = self.coords[:, None, :] - self.coords[None, :, :]  # (N+1,N+1,2)
        self.dist_matrix = np.sqrt((diff**2).sum(-1)).astype(np.float32)

        # Scalar parameters
        self.Q_t = float(raw_instance["truck_capacity"])
        self.Q_d = float(raw_instance["drone_capacity"])
        self.T_max = float(raw_instance["system_duration"])
        self.D_max = float(raw_instance["drone_trip_duration"])
        self.v_t = float(raw_instance["truck_speed"])
        self.v_d = float(raw_instance["drone_speed"])
        self.c_t = float(raw_instance["truck_cost"])
        self.c_d = float(raw_instance["drone_cost"])
        self.launch_time = float(raw_instance["launch_time"])  # lift-off overhead
        self.land_time = float(raw_instance["land_time"])  # landing/securing overhead
        self.service_time = float(raw_instance["service_time"])

    # ------------------------------------------------------------------
    # 2. initial_state
    # ------------------------------------------------------------------

    def initial_state(self) -> VRPBTWState:
        K = self.K
        depot = self.coords[DEPOT]

        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True  # depot never needs serving

        return VRPBTWState(
            # Trucks start at depot, full capacity, time=0
            truck_pos=np.tile(depot, (K, 1)).astype(np.float32),
            truck_load=np.full(K, self.Q_t, dtype=np.float32),
            truck_time=np.zeros(K, dtype=np.float32),
            truck_dist=np.zeros(K, dtype=np.float32),
            truck_route=[[DEPOT] for _ in range(K)],
            # Drones start co-located with trucks (at depot), idle
            drone_pos=np.tile(depot, (K, 1)).astype(np.float32),
            drone_load=np.full(K, self.Q_d, dtype=np.float32),
            drone_time=np.zeros(K, dtype=np.float32),
            drone_dist=np.zeros(K, dtype=np.float32),
            drone_airtime=np.zeros(K, dtype=np.float32),
            drone_active=np.zeros(K, dtype=bool),
            drone_rendezvous=np.full(K, -1, dtype=np.int32),
            drone_route=[[] for _ in range(K)],
            served=served,
            step=0,
        )

    # ------------------------------------------------------------------
    # 3. get_action_mask
    # ------------------------------------------------------------------

    def get_action_mask(self, state: VRPBTWState) -> ActionMask:
        """
        Flat action = fleet_id * (2*(N+1))  +  vehicle_id * (N+1)  +  node_id

        node_id = 0         → return to depot (only when all customers served
                               or vehicle must return due to constraints)
        node_id = 1..N      → serve customer node_id
        """
        N1 = self.n_customers + 1  # number of nodes (depot + customers)
        size = self.K * 2 * N1
        mask = np.zeros(size, dtype=bool)

        for k in range(self.K):
            for v in (TRUCK, DRONE):
                for j in range(N1):
                    flat = k * (2 * N1) + v * N1 + j
                    mask[flat] = self._is_feasible(state, k, v, j)

        return ActionMask.from_bool_array(mask)

    def _is_feasible(self, state: VRPBTWState, k: int, v: int, j: int) -> bool:
        """Return True if fleet k, vehicle v can visit node j next."""
        # depot return is only valid if no unserved customers remain
        if j == DEPOT:
            return not state.served[1:].all() == False  # always allow depot return
        # Customer already served
        if state.served[j]:
            return False

        if v == TRUCK:
            return self._truck_feasible(state, k, j)
        else:
            return self._drone_feasible(state, k, j)

    def _truck_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check all truck constraints for visiting customer j."""
        # Capacity
        if state.truck_load[k] < self.demands[j]:
            return False

        dist = self.dist_matrix[self._truck_node(state, k), j]
        travel = dist / self.v_t
        arrive = state.truck_time[k] + travel

        # Arrive before time window closes
        if arrive > self.tw_close[j]:
            return False

        # Depart (after service) within system duration
        depart = max(arrive, self.tw_open[j]) + self.service_time
        if depart > self.T_max:
            return False

        # Must be able to return to depot after service
        return_dist = self.dist_matrix[j, DEPOT]
        return_arrive = depart + return_dist / self.v_t
        if return_arrive > self.T_max:
            return False

        return True

    def _drone_feasible(self, state: VRPBTWState, k: int, j: int) -> bool:
        """Check all drone constraints for a sortie to customer j."""
        # Drone must be idle (not already in flight)
        if state.drone_active[k]:
            return False

        # Drone must be co-located with the truck (launch from truck)
        if not np.allclose(state.drone_pos[k], state.truck_pos[k], atol=1e-4):
            return False

        # Capacity
        if state.drone_load[k] < self.demands[j]:
            return False

        # --- Compute drone sortie timing ---
        # Phase 1: launch_time (overhead before take-off)
        # Phase 2: fly truck_pos → j  at drone_speed
        # Phase 3: service_time at j
        # Phase 4: fly j → rendezvous at drone_speed
        # Phase 5: land_time (overhead after landing)
        # Rendezvous = depot (conservative; truck may progress further)

        dist_out = self.dist_matrix[self._truck_node(state, k), j]
        dist_back = self.dist_matrix[j, DEPOT]  # worst-case rendezvous

        airborne_time = (dist_out + dist_back) / self.v_d
        total_sortie = (
            self.launch_time + airborne_time + self.service_time + self.land_time
        )

        # Drone trip duration limit (battery)
        if airborne_time > self.D_max:
            return False

        depart_time = state.drone_time[k]
        arrive_j = depart_time + self.launch_time + dist_out / self.v_d

        # Time window at customer
        if arrive_j > self.tw_close[j]:
            return False

        # System duration
        finish = depart_time + total_sortie
        if finish > self.T_max:
            return False

        return True

    def _truck_node(self, state: VRPBTWState, k: int) -> int:
        """Return the index of the last node the truck visited."""
        route = state.truck_route[k]
        if not route:
            return DEPOT
        # Find node index from coordinates
        for node_idx in range(self.n_customers + 1):
            if np.allclose(self.coords[node_idx], state.truck_pos[k], atol=1e-4):
                return node_idx
        return DEPOT

    # ------------------------------------------------------------------
    # 4. apply_action
    # ------------------------------------------------------------------

    def apply_action(self, state: VRPBTWState, action: int) -> StepResult:
        """
        Decode flat action → (fleet k, vehicle v, node j) and apply it.
        Returns the next state, dense reward, termination flags, and new mask.
        """
        N1 = self.n_customers + 1
        k = action // (2 * N1)
        v = (action // N1) % 2
        j = action % N1

        # Deep-copy mutable state fields
        state = self._copy_state(state)

        reward = 0.0

        if v == TRUCK:
            reward = self._apply_truck(state, k, j)
        else:
            reward = self._apply_drone(state, k, j)

        state.step += 1

        terminated = self._check_termination(state)
        if terminated:
            reward += self._terminal_reward(state)

        next_mask = (
            self.get_action_mask(state)
            if not terminated
            else ActionMask.all_valid(self.K * 2 * N1)
        )

        # Safety: if no feasible actions remain but not terminated, force terminate
        if not terminated and next_mask.is_empty():
            terminated = True
            reward += self._terminal_reward(state)

        info = {
            "fleet": k,
            "vehicle": "truck" if v == TRUCK else "drone",
            "node": j,
            "served_count": int(state.served[1:].sum()),
        }
        return StepResult(state, reward, terminated, False, next_mask, info)

    def _apply_truck(self, state: VRPBTWState, k: int, j: int) -> float:
        """Move truck k to node j; update load, time, distance. Return reward."""
        from_node = self._truck_node(state, k)
        dist = self.dist_matrix[from_node, j]
        travel = dist / self.v_t

        arrive = state.truck_time[k] + travel
        serve_start = max(arrive, self.tw_open[j])

        # Time-window bonus: reward earlier arrivals inside window
        tw_bonus = 0.0
        if arrive <= self.tw_close[j]:
            tw_bonus = 0.1 * (self.tw_close[j] - arrive) / (self.T_max + 1e-6)

        state.truck_time[k] = serve_start + self.service_time
        state.truck_pos[k] = self.coords[j].copy()
        state.truck_dist[k] += dist
        state.truck_load[k] -= self.demands[j]
        state.truck_route[k].append(j)

        if j != DEPOT:
            state.served[j] = True

        # If drone was waiting for this truck node as rendezvous, sync
        if state.drone_rendezvous[k] == j:
            state.drone_time[k] = max(state.drone_time[k], state.truck_time[k])
            state.drone_pos[k] = self.coords[j].copy()
            state.drone_active[k] = False
            state.drone_rendezvous[k] = -1
            state.drone_airtime[k] = 0.0

        # Dense reward: negative distance cost
        reward = -(self.c_t * dist) + tw_bonus
        return float(reward)

    def _apply_drone(self, state: VRPBTWState, k: int, j: int) -> float:
        """
        Dispatch drone k on a sortie to customer j.

        Timing breakdown (all additive):
          launch_time   → lift-off overhead (separate parameter)
          dist_out/v_d  → fly from truck position to customer j
          service_time  → serve customer j
          dist_back/v_d → fly from j to rendezvous node (next truck stop or depot)
          land_time     → landing/securing overhead (separate parameter)

        The rendezvous node is set to the depot conservatively; in a full
        implementation it would be the truck's next planned stop.
        """
        from_node = self._truck_node(state, k)
        dist_out = self.dist_matrix[from_node, j]
        dist_back = self.dist_matrix[j, DEPOT]  # rendezvous at depot (conservative)
        rendezvous = DEPOT

        depart_time = state.drone_time[k]

        # Full sortie timeline
        t_launch = depart_time + self.launch_time  # lift-off complete
        t_arrive_j = t_launch + dist_out / self.v_d  # arrive at customer
        t_depart_j = max(t_arrive_j, self.tw_open[j]) + self.service_time
        t_arrive_rv = t_depart_j + dist_back / self.v_d  # arrive at rendezvous
        t_landed = t_arrive_rv + self.land_time  # fully secured on truck

        airborne = (dist_out + dist_back) / self.v_d  # pure flight time

        # Update drone state
        state.drone_time[k] = t_landed
        state.drone_pos[k] = self.coords[rendezvous].copy()
        state.drone_dist[k] += dist_out + dist_back
        state.drone_load[k] -= self.demands[j]
        state.drone_active[k] = True  # in-flight until rendezvous
        state.drone_airtime[k] = airborne
        state.drone_rendezvous[k] = rendezvous
        state.drone_route[k].append(j)

        if j != DEPOT:
            state.served[j] = True

        # Time-window bonus
        tw_bonus = 0.0
        if t_arrive_j <= self.tw_close[j]:
            tw_bonus = 0.1 * (self.tw_close[j] - t_arrive_j) / (self.T_max + 1e-6)

        # Dense reward: negative distance cost (out + back)
        reward = -(self.c_d * (dist_out + dist_back)) + tw_bonus
        return float(reward)

    # ------------------------------------------------------------------
    # Terminal logic
    # ------------------------------------------------------------------

    def _check_termination(self, state: VRPBTWState) -> bool:
        """Episode ends when all customers served OR no feasible moves remain."""
        return bool(state.served[1:].all())

    def _terminal_reward(self, state: VRPBTWState) -> float:
        """
        Large positive bonus if all customers served.
        Penalty proportional to unserved count otherwise.
        """
        unserved = int((~state.served[1:]).sum())
        if unserved == 0:
            return 10.0  # all customers served bonus
        return -2.0 * unserved  # penalty per unserved customer

    # ------------------------------------------------------------------
    # 5. state_to_obs  →  (N+1, NODE_FEAT_DIM)
    # ------------------------------------------------------------------

    def state_to_obs(self, state: VRPBTWState) -> np.ndarray:
        """
        Per-node feature matrix of shape (N+1, NODE_FEAT_DIM).

        Features per node i:
          0: x / max_coord
          1: y / max_coord
          2: tw_open[i] / T_max
          3: tw_close[i] / T_max
          4: demand[i] / Q_t
          5: served[i] (float)
          6: min distance to any truck / max_dist
          7: min distance to any drone / max_dist
          8: time slack = (tw_close[i] - min_truck_time) / T_max
          9: is_depot flag
        """
        N1 = self.n_customers + 1
        max_coord = self.coords.max() + 1e-6
        max_dist = self.dist_matrix.max() + 1e-6

        obs = np.zeros((N1, NODE_FEAT_DIM), dtype=np.float32)

        # Nearest truck / drone distances to each node
        min_truck_dist = np.min(
            np.linalg.norm(
                self.coords[:, None, :] - state.truck_pos[None, :, :], axis=2
            ),
            axis=1,
        )  # (N+1,)
        min_drone_dist = np.min(
            np.linalg.norm(
                self.coords[:, None, :] - state.drone_pos[None, :, :], axis=2
            ),
            axis=1,
        )  # (N+1,)

        min_truck_time = state.truck_time.min()

        for i in range(N1):
            slack = (self.tw_close[i] - min_truck_time) / (self.T_max + 1e-6)
            obs[i] = [
                self.coords[i, 0] / max_coord,
                self.coords[i, 1] / max_coord,
                self.tw_open[i] / (self.T_max + 1e-6),
                self.tw_close[i] / (self.T_max + 1e-6),
                self.demands[i] / (self.Q_t + 1e-6),
                float(state.served[i]),
                min_truck_dist[i] / max_dist,
                min_drone_dist[i] / max_dist,
                float(np.clip(slack, -1.0, 1.0)),
                float(i == DEPOT),
            ]
        return obs

    # ------------------------------------------------------------------
    # 6. evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: VRPBTWState) -> float:
        """
        Objective = −total operational cost (higher is better).
        Cost = Σ_k (truck_cost * truck_dist[k] + drone_cost * drone_dist[k])
        Penalty for unserved customers.
        """
        truck_cost = float((state.truck_dist * self.c_t).sum())
        drone_cost = float((state.drone_dist * self.c_d).sum())
        unserved = int((~state.served[1:]).sum())
        penalty = unserved * 1000.0  # heavy penalty per unserved customer
        return -(truck_cost + drone_cost + penalty)

    # ------------------------------------------------------------------
    # 7. is_complete
    # ------------------------------------------------------------------

    def is_complete(self, state: VRPBTWState) -> bool:
        return bool(state.served[1:].all())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        return self.K * 2 * (self.n_customers + 1)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.n_customers + 1, NODE_FEAT_DIM)

    # ------------------------------------------------------------------
    # Optional overrides
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
                "unserved": int((~state.served[1:]).sum()),
            },
        )

    def heuristic_solution(self) -> Optional[float]:
        """
        Nearest-neighbour heuristic (truck only, no drone) as a lower bound
        for reward shaping baseline.
        """
        served = np.zeros(self.n_customers + 1, dtype=bool)
        served[DEPOT] = True
        current = DEPOT
        time = 0.0
        dist = 0.0
        load = self.Q_t

        while not served[1:].all():
            best_j, best_d = -1, float("inf")
            for j in range(1, self.n_customers + 1):
                if served[j] or self.demands[j] > load:
                    continue
                d = self.dist_matrix[current, j]
                arrive = time + d / self.v_t
                if arrive > self.tw_close[j]:
                    continue
                if d < best_d:
                    best_d, best_j = d, j
            if best_j == -1:
                break  # no feasible next customer
            d = self.dist_matrix[current, best_j]
            time += d / self.v_t
            time = max(time, self.tw_open[best_j]) + self.service_time
            dist += d
            load -= self.demands[best_j]
            served[best_j] = True
            current = best_j

        dist += self.dist_matrix[current, DEPOT]
        unserved = int((~served[1:]).sum())
        return -(dist * self.c_t + unserved * 1000.0)

    def instance_features(self) -> np.ndarray:
        """
        Global context vector injected into the decoder.
        Shape: (K * 6,)  — per-fleet truck+drone summary.
        """
        # This is called once per observation; we need the current state.
        # Since instance_features() has no state argument in the base class,
        # we return zeros here and embed fleet state directly in state_to_obs.
        return np.zeros(self.K * 6, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_state(s: VRPBTWState) -> VRPBTWState:
        return VRPBTWState(
            truck_pos=s.truck_pos.copy(),
            truck_load=s.truck_load.copy(),
            truck_time=s.truck_time.copy(),
            truck_dist=s.truck_dist.copy(),
            truck_route=[list(r) for r in s.truck_route],
            drone_pos=s.drone_pos.copy(),
            drone_load=s.drone_load.copy(),
            drone_time=s.drone_time.copy(),
            drone_dist=s.drone_dist.copy(),
            drone_airtime=s.drone_airtime.copy(),
            drone_active=s.drone_active.copy(),
            drone_rendezvous=s.drone_rendezvous.copy(),
            drone_route=[list(r) for r in s.drone_route],
            served=s.served.copy(),
            step=s.step,
        )


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------


def generate_vrpbtw(
    n_customers: int = 10,
    n_fleets: int = 2,
    grid_size: float = 100.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Generate a random VRPBTW instance.

    Time windows are set so roughly 80% of customers have feasible windows
    given a truck travelling at v_t=1.0 from the depot.
    """
    rng = np.random.default_rng(seed)

    depot = (grid_size / 2.0, grid_size / 2.0)

    coords = rng.uniform(0, grid_size, (n_customers, 2))
    demands = rng.uniform(1, 10, n_customers).round(1)

    # Estimate travel time from depot for each customer
    depot_arr = np.array(depot)
    dist_from_depot = np.linalg.norm(coords - depot_arr, axis=1)
    earliest_arrive = dist_from_depot / 1.0  # truck_speed=1.0

    # Time windows: open slightly before earliest arrival, close after
    tw_open = np.maximum(0.0, earliest_arrive - rng.uniform(5, 15, n_customers))
    tw_close = earliest_arrive + rng.uniform(20, 50, n_customers)

    system_duration = float(tw_close.max() + 30.0)
    drone_trip_duration = grid_size * 0.6  # generous battery

    customers = np.column_stack([coords, tw_open, tw_close, demands]).tolist()

    return {
        "depot": list(depot),
        "customers": customers,
        "n_fleets": n_fleets,
        "truck_capacity": 50.0,
        "drone_capacity": 15.0,
        "system_duration": system_duration,
        "drone_trip_duration": drone_trip_duration,
        "truck_speed": 1.0,
        "drone_speed": 2.0,  # drones are faster
        "truck_cost": 1.0,
        "drone_cost": 0.5,  # drones cheaper per distance
        "launch_time": 2.0,  # time to lift off (separate)
        "land_time": 3.0,  # time to land & secure (separate)
        "service_time": 5.0,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo():
    from environments.combinatorial_env import CombinatorialEnv

    problem = VRPBTWProblem(n_customers=5, n_fleets=1)
    env = CombinatorialEnv(problem, max_steps=50, dense_shaping=True)

    raw = generate_vrpbtw(n_customers=5, n_fleets=1, seed=0)
    obs, info = env.reset(raw)

    print("VRPBTW Demo — Random Feasible Policy")
    print(f"  Customers     : {problem.n_customers}")
    print(f"  Fleets        : {problem.n_fleets}")
    print(f"  Obs shape     : {obs.shape}")
    print(f"  Action space  : {problem.action_space_size}")
    print(f"  launch_time   : {problem.launch_time}")
    print(f"  land_time     : {problem.land_time}")

    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        feasible = info["feasible_actions"]
        if len(feasible) == 0:
            break
        action = int(np.random.choice(feasible))
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        steps += 1
        done = terminated or truncated

    sol = env.decode_current_solution()
    print(f"\n  Steps taken   : {steps}")
    print(f"  Episode reward: {total_reward:.3f}")
    print(f"\n{sol.summary()}")
    print(f"\n  Truck routes  : {sol.metadata['truck_routes']}")
    print(f"  Drone routes  : {sol.metadata['drone_routes']}")
    print(f"  Served        : {sol.metadata['served_count']} / {problem.n_customers}")
    print(f"  Unserved      : {sol.metadata['unserved']}")


if __name__ == "__main__":
    demo()
