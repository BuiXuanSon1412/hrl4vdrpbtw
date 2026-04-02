import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TSPDState:
    coords: torch.Tensor          # [B, N, 2] - node coordinates
    demand: torch.Tensor          # [B, N] - customer demand (1=unserved, 0=served)
    truck_loc: torch.Tensor       # [B] - current truck node index
    drone_loc: torch.Tensor       # [B] - current drone node index
    drone_status: torch.Tensor    # [B] - s_d: 0=available, 1=in service
    drone_returned: torch.Tensor  # [B] - r_d: 0=not returned, 1=returned
    truck_action: torch.Tensor    # [B, 2] - (target_node, remaining_time)
    drone_action: torch.Tensor    # [B, 2] - (target_node, remaining_time)
    current_time: torch.Tensor    # [B] - current elapsed time


class TSPDEnvironment:
    def __init__(self,n_nodes: int = 20,batch_size: int = 128,truck_speed: float = 1.0,drone_speed_ratio: float = 2.0,device: str = "cpu",):
        self.n_nodes = n_nodes          
        self.n_customers = n_nodes - 1
        self.batch_size = batch_size
        self.truck_speed = truck_speed
        self.drone_speed = truck_speed * drone_speed_ratio
        self.device = torch.device(device)
        self.depot_idx = 0              

    def reset(self, instances: Optional[torch.Tensor] = None) -> TSPDState:
        B = self.batch_size

        if instances is not None:
            coords = instances.to(self.device)
            B = coords.shape[0]
        else:
            coords = torch.rand(B, self.n_nodes, 2, device=self.device)

        # Initial state: all customers unserved
        demand = torch.ones(B, self.n_nodes, device=self.device)
        demand[:, self.depot_idx] = 0.0  # Depot has no demand

        # Both vehicles start at depot
        truck_loc = torch.zeros(B, dtype=torch.long, device=self.device)
        drone_loc = torch.zeros(B, dtype=torch.long, device=self.device)

        # Drone available, not in flight
        drone_status = torch.zeros(B, device=self.device)
        drone_returned = torch.zeros(B, device=self.device)

        # No pending actions
        truck_action = torch.zeros(B, 2, device=self.device)
        drone_action = torch.zeros(B, 2, device=self.device)
        # Initialize pending action targets to depot
        truck_action[:, 0] = self.depot_idx
        drone_action[:, 0] = self.depot_idx

        current_time = torch.zeros(B, device=self.device)

        self.state = TSPDState(
            coords=coords,
            demand=demand,
            truck_loc=truck_loc,
            drone_loc=drone_loc,
            drone_status=drone_status,
            drone_returned=drone_returned,
            truck_action=truck_action,
            drone_action=drone_action,
            current_time=current_time,
        )
        self.batch_size = B
        return self.state

    def _euclidean_dist(self, coords: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance between node i and j for each batch."""
        ci = coords[torch.arange(coords.shape[0]), i]  # [B, 2]
        cj = coords[torch.arange(coords.shape[0]), j]  # [B, 2]
        return torch.norm(ci - cj, dim=-1)              # [B]

    def get_action_mask(self, state: TSPDState) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = state.demand.shape
        device = self.device

        served_mask = (state.demand == 0)
        served_mask[:, self.depot_idx] = False  

        truck_mask = served_mask.clone()
        drone_mask = served_mask.clone()

        # Truck with pending action must continue
        truck_pending = state.truck_action[:, 1] > 0  # [B]
        # Drone with pending action must continue
        drone_pending = state.drone_action[:, 1] > 0  # [B]

        for b in range(B):
            if truck_pending[b]:
                # Must continue to truck's target
                target = state.truck_action[b, 0].long()
                truck_mask[b] = True
                truck_mask[b, target] = False

            if drone_pending[b]:
                # Must continue to drone's target
                target = state.drone_action[b, 0].long()
                drone_mask[b] = True
                drone_mask[b, target] = False
            else:
                # Drone in service must return to truck's current/next location
                if state.drone_status[b] == 1:
                    drone_mask[b] = True
                    drone_mask[b, state.truck_loc[b]] = False

                # If drone returned, it cannot relaunch until truck visits another customer
                if state.drone_returned[b] == 1:
                    drone_mask[b] = True
                    # Allow drone to stay at depot (no-op)
                    drone_mask[b, self.depot_idx] = False

        return truck_mask, drone_mask

    def step(
        self, state: TSPDState, truck_action: torch.Tensor, drone_action: torch.Tensor
    ) -> Tuple[TSPDState, torch.Tensor, torch.Tensor]:
        B = state.coords.shape[0]
        device = self.device

        coords = state.coords
        demand = state.demand.clone()

        # Compute travel times
        # Truck: distance / truck_speed
        truck_dist = self._euclidean_dist(coords, state.truck_loc, truck_action)
        truck_time = truck_dist / self.truck_speed  # [B]

        # Drone: distance / drone_speed
        drone_dist = self._euclidean_dist(coords, state.drone_loc, drone_action)
        drone_time = drone_dist / self.drone_speed  # [B]

        # Account for pending action residuals
        truck_remaining = state.truck_action[:, 1]
        drone_remaining = state.drone_action[:, 1]

        # Δt = min of all arrival times (Eq. in Section 3.2 transition)
        effective_truck = torch.where(truck_remaining > 0, truck_remaining, truck_time)
        effective_drone = torch.where(drone_remaining > 0, drone_remaining, drone_time)

        delta_t = torch.min(effective_truck, effective_drone)  # [B]

        # Reward: r(s,a) = -Δt  (Eq. 2)
        reward = -delta_t

        # Update current time
        new_time = state.current_time + delta_t

        # Update positions
        new_truck_loc = state.truck_loc.clone()
        new_drone_loc = state.drone_loc.clone()

        truck_arrived = effective_truck <= delta_t + 1e-6
        drone_arrived = effective_drone <= delta_t + 1e-6

        new_truck_loc = torch.where(truck_arrived, truck_action, state.truck_loc)
        new_drone_loc = torch.where(drone_arrived, drone_action, state.drone_loc)

        # Update demand: if truck or drone arrives at a customer
        for b in range(B):
            if truck_arrived[b] and truck_action[b] != self.depot_idx:
                demand[b, truck_action[b]] = 0
            if drone_arrived[b] and drone_action[b] != self.depot_idx:
                demand[b, drone_action[b]] = 0

        # Update drone status
        new_drone_status = state.drone_status.clone()
        new_drone_returned = state.drone_returned.clone()

        for b in range(B):
            if drone_arrived[b]:
                if drone_action[b] != self.depot_idx and drone_action[b] != state.truck_loc[b]:
                    # Drone in service (delivering to customer)
                    new_drone_status[b] = 1
                else:
                    # Drone returned to truck
                    new_drone_status[b] = 0
                    new_drone_returned[b] = 1 if drone_arrived[b] else 0

        # Update pending actions
        new_truck_action = state.truck_action.clone()
        new_drone_action = state.drone_action.clone()

        # For truck: (target, max(0, remaining - delta_t))
        new_truck_remaining = torch.clamp(effective_truck - delta_t, min=0)
        new_drone_remaining = torch.clamp(effective_drone - delta_t, min=0)

        new_truck_action[:, 0] = truck_action
        new_truck_action[:, 1] = new_truck_remaining

        new_drone_action[:, 0] = drone_action
        new_drone_action[:, 1] = new_drone_remaining

        # Check done: all customer demands satisfied
        done = (demand[:, 1:].sum(dim=-1) < 1e-6)  # [B]

        next_state = TSPDState(
            coords=coords,
            demand=demand,
            truck_loc=new_truck_loc,
            drone_loc=new_drone_loc,
            drone_status=new_drone_status,
            drone_returned=new_drone_returned,
            truck_action=new_truck_action,
            drone_action=new_drone_action,
            current_time=new_time,
        )

        return next_state, reward, done

    def get_static_features(self, state: TSPDState) -> torch.Tensor:
        return state.coords

    def get_dynamic_features(self, state: TSPDState) -> torch.Tensor:
        B, N = state.demand.shape
        device = self.device

        # Demand feature
        demand_feat = state.demand.unsqueeze(-1)  # [B, N, 1]

        # Truck position one-hot
        truck_pos = torch.zeros(B, N, device=device)
        truck_pos.scatter_(1, state.truck_loc.unsqueeze(1), 1.0)
        truck_pos = truck_pos.unsqueeze(-1)  # [B, N, 1]

        # Drone position one-hot
        drone_pos = torch.zeros(B, N, device=device)
        drone_pos.scatter_(1, state.drone_loc.unsqueeze(1), 1.0)
        drone_pos = drone_pos.unsqueeze(-1)  # [B, N, 1]

        # Depot indicator
        depot_feat = torch.zeros(B, N, 1, device=device)
        depot_feat[:, self.depot_idx] = 1.0

        return torch.cat([demand_feat, truck_pos, drone_pos, depot_feat], dim=-1)  # [B, N, 4]


def generate_tspd_instances(n_instances: int,n_nodes: int,instance_type: str = "random",device: str = "cpu",) -> torch.Tensor:
    device = torch.device(device)

    if instance_type == "random":
        # Depot at lower-left corner, customers uniform in [0,1]^2
        coords = torch.rand(n_instances, n_nodes, 2, device=device)
        # Force depot to lower-left region
        coords[:, 0, :] = torch.rand(n_instances, 2, device=device) * 0.1
    elif instance_type == "uniform":
        coords = torch.rand(n_instances, n_nodes, 2, device=device)
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")

    return coords