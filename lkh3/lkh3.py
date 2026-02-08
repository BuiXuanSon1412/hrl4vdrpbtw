import numpy as np
import json
import argparse
import sys
import os
import time
import random
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Node:
    """Represents a customer or depot node"""
    id: int
    coord: np.ndarray
    demand: int
    node_type: str  
    time_window: Tuple[float, float]
    service_time: float


@dataclass
class Solution:
    """Represents a complete solution"""
    truck_routes: List[List[int]] 
    drone_assignments: Dict[int, List[List[int]]]  
    waiting_time: float
    total_cost: float
    feasible: bool

class CTDTWB_Instance:
    def __init__(self, data_path: str):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"JSON not found: {data_path}")
        self.data_path = data_path
        self.filename = os.path.basename(data_path)
        with open(data_path, "r") as f:
            data = json.load(f)
        
        config = data['Config']
        nodes_data = data['Nodes']
        
        # General parameters
        self.num_customers = config['General']['NUM_CUSTOMERS']
        self.max_coord = config['General']['MAX_COORD_KM']
        self.t_max_system = config['General']['T_MAX_SYSTEM_H']
        
        # Vehicle parameters
        self.num_trucks = config['Vehicles']['NUM_TRUCKS']
        self.num_drones = config['Vehicles']['NUM_DRONES']
        self.v_truck = config['Vehicles']['V_TRUCK_KM_H']
        self.v_drone = config['Vehicles']['V_DRONE_KM_H']
        self.q_truck = config['Vehicles']['CAPACITY_TRUCK']
        self.q_drone = config['Vehicles']['CAPACITY_DRONE']
        self.tau_l = config['Vehicles']['DRONE_TAKEOFF_MIN'] / 60.0
        self.tau_r = config['Vehicles']['DRONE_LANDING_MIN'] / 60.0
        self.service_time = config['Vehicles']['SERVICE_TIME_MIN'] / 60.0
        self.t_max_drone = config['Vehicles']['DRONE_DURATION_H']
        
        # Depot
        depot_info = config['Depot']
        depot_coord = np.array(depot_info['coord'])
        depot_tw = tuple(depot_info['time_window_h'])
        
        # Build nodes
        self.nodes = [Node(
            id=0,
            coord=depot_coord,
            demand=0,
            node_type='DEPOT',
            time_window=depot_tw,
            service_time=0.0
        )]
        
        self.linehaul_nodes = []
        self.backhaul_nodes = []
        
        for node_data in nodes_data:
            node = Node(
                id=node_data['id'],
                coord=np.array(node_data['coord']),
                demand=node_data['demand'],
                node_type=node_data['type'],
                time_window=tuple(node_data['tw_h']),
                service_time=self.service_time
            )
            self.nodes.append(node)
            
            if node.node_type == 'LINEHAUL':
                self.linehaul_nodes.append(node.id)
            else:
                self.backhaul_nodes.append(node.id)
        
        # Calculate distance matrices
        n = len(self.nodes)
        self.dist_truck = np.zeros((n, n))
        self.dist_drone = np.zeros((n, n))
        self.time_truck = np.zeros((n, n))
        self.time_drone = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Manhattan distance for truck
                    self.dist_truck[i, j] = np.sum(np.abs(
                        self.nodes[i].coord - self.nodes[j].coord
                    ))
                    # Euclidean distance for drone
                    self.dist_drone[i, j] = np.linalg.norm(
                        self.nodes[i].coord - self.nodes[j].coord
                    )
                    
                    self.time_truck[i, j] = self.dist_truck[i, j] / self.v_truck
                    self.time_drone[i, j] = self.dist_drone[i, j] / self.v_drone
        self.customers = list(range(1, len(self.nodes)))


class LKH3_CTDTWB:    
    def __init__(self, instance: CTDTWB_Instance, seed: int = 42):
        self.instance = instance
        self.best_solution = None
        self.best_waiting_time = float('inf')
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_initial_solution(self) -> Solution:
        solution = Solution(
            truck_routes=[],
            drone_assignments={},
            waiting_time=0.0,
            total_cost=0.0,
            feasible=True
        )
        
        unvisited = set(self.instance.customers)
        for truck_id in range(1, self.instance.num_trucks + 1):
            if not unvisited:
                break
            
            route = [0]  
            current_pos = 0
            current_time = 0.0
            current_load = 0
            
            while unvisited:
                best_next = None
                best_score = float('inf')
                
                for customer in unvisited:
                    node = self.instance.nodes[customer]
                    new_load = current_load + node.demand
                    if new_load > self.instance.q_truck:
                        continue
                    arrival_time = current_time + self.instance.time_truck[current_pos, customer]
                    if arrival_time > node.time_window[1] + 20: 
                        continue
                    score = self.instance.dist_truck[current_pos, customer]
                    score += max(0, node.time_window[0] - arrival_time) * 10
                    
                    if score < best_score:
                        best_score = score
                        best_next = customer
                
                if best_next is None:
                    break
                route.append(best_next)
                unvisited.remove(best_next)
                node = self.instance.nodes[best_next]
                arrival_time = current_time + self.instance.time_truck[current_pos, best_next]
                service_start = max(arrival_time, node.time_window[0])
                current_time = service_start + node.service_time
                current_load += node.demand
                current_pos = best_next
            
            if len(route) > 1:
                route.append(0)  # Return to depot
                solution.truck_routes.append(route)
                solution.drone_assignments[truck_id] = []
        
        if unvisited:
            print(f"Warning: {len(unvisited)} customers not assigned. Force assigning...")
            # Create additional routes if needed
            while unvisited:
                route = [0]
                customers_in_route = list(unvisited)[:min(5, len(unvisited))]  
                route.extend(customers_in_route)
                route.append(0)
                
                solution.truck_routes.append(route)
                for c in customers_in_route:
                    unvisited.remove(c)
                
                truck_id = len(solution.truck_routes)
                solution.drone_assignments[truck_id] = []
        
        return solution
    
    def evaluate_solution(self, solution: Solution):
        max_waiting = 0.0
        total_cost = 0.0
        feasible = True
        served_customers = set()
        
        for truck_id, route in enumerate(solution.truck_routes, start=1):
            current_time = 0.0
            current_load = 0
            
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                
                current_time += self.instance.time_truck[from_node, to_node]
                total_cost += self.instance.dist_truck[from_node, to_node] * 25.0
                
                if to_node == 0:  
                    continue
                
                served_customers.add(to_node)
                node = self.instance.nodes[to_node]
                service_start = max(current_time, node.time_window[0])
                waiting = max(0, service_start + node.service_time - node.time_window[1])
                max_waiting = max(max_waiting, waiting)
                
                # Check if late arrival is too severe
                if current_time > node.time_window[1] + 10:
                    feasible = False
                
                # Update time and load
                current_time = service_start + node.service_time
                current_load += node.demand
                
                # Check capacity - allow negative for backhaul
                if current_load > self.instance.q_truck:
                    feasible = False
        
        # Check if all customers are served
        all_customers = set(self.instance.customers)
        if served_customers != all_customers:
            unserved = all_customers - served_customers
            print(f"Warning: {len(unserved)} customers not served: {unserved}")
            feasible = False
            # Penalty for unserved customers
            max_waiting += len(unserved) * 100.0
        
        # Add fixed cost per truck
        total_cost += len(solution.truck_routes) * 500.0
        
        solution.waiting_time = max_waiting
        solution.total_cost = total_cost
        solution.feasible = feasible
    
    def perturb_solution(self, solution: Solution) -> Solution:
        new_solution = deepcopy(solution)
        
        if not new_solution.truck_routes:
            return new_solution
        
        operator = random.choice(['2-opt', 'swap', 'relocate', 'cross-exchange'])
        
        if operator == '2-opt':
            new_solution = self.two_opt(new_solution)
        elif operator == 'swap':
            new_solution = self.swap_operator(new_solution)
        elif operator == 'relocate':
            new_solution = self.relocate_operator(new_solution)
        elif operator == 'cross-exchange':
            new_solution = self.cross_exchange(new_solution)
        
        return new_solution
    
    def two_opt(self, solution: Solution) -> Solution:
        if not solution.truck_routes:
            return solution
        
        route_idx = random.randint(0, len(solution.truck_routes) - 1)
        route = solution.truck_routes[route_idx]
        
        if len(route) <= 3:
            return solution
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        solution.truck_routes[route_idx] = new_route
        
        return solution
    
    def swap_operator(self, solution: Solution) -> Solution:
        if not solution.truck_routes:
            return solution
        
        if len(solution.truck_routes) == 1 and len(solution.truck_routes[0]) <= 3:
            return solution
        route_idx1 = random.randint(0, len(solution.truck_routes) - 1)
        route_idx2 = random.randint(0, len(solution.truck_routes) - 1)
        route1 = solution.truck_routes[route_idx1]
        route2 = solution.truck_routes[route_idx2]
        
        if len(route1) <= 2 or len(route2) <= 2:
            return solution
        pos1 = random.randint(1, len(route1) - 2)
        pos2 = random.randint(1, len(route2) - 2)
        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        return solution
    
    def relocate_operator(self, solution: Solution) -> Solution:
        if len(solution.truck_routes) < 2:
            return solution
        
        route_idx1 = random.randint(0, len(solution.truck_routes) - 1)
        route_idx2 = random.randint(0, len(solution.truck_routes) - 1)
        
        if route_idx1 == route_idx2:
            return solution
        
        route1 = solution.truck_routes[route_idx1]
        route2 = solution.truck_routes[route_idx2]
        
        if len(route1) <= 2:
            return solution
        pos1 = random.randint(1, len(route1) - 2)
        customer = route1.pop(pos1)
        
        if len(route2) <= 2:
            insert_pos = 1
        else:
            insert_pos = random.randint(1, len(route2) - 1)
        
        route2.insert(insert_pos, customer)
        
        return solution
    
    def cross_exchange(self, solution: Solution) -> Solution:
        if len(solution.truck_routes) < 2:
            return solution
        route_idx1 = random.randint(0, len(solution.truck_routes) - 1)
        route_idx2 = random.randint(0, len(solution.truck_routes) - 1)
        
        if route_idx1 == route_idx2:
            return solution
        
        route1 = solution.truck_routes[route_idx1]
        route2 = solution.truck_routes[route_idx2]
        
        if len(route1) <= 3 or len(route2) <= 3:
            return solution
        
        # Select segments
        start1 = random.randint(1, len(route1) - 3)
        end1 = random.randint(start1, len(route1) - 2)
        
        start2 = random.randint(1, len(route2) - 3)
        end2 = random.randint(start2, len(route2) - 2)
        
        # Exchange segments
        seg1 = route1[start1:end1+1]
        seg2 = route2[start2:end2+1]
        
        new_route1 = route1[:start1] + seg2 + route1[end1+1:]
        new_route2 = route2[:start2] + seg1 + route2[end2+1:]
        
        solution.truck_routes[route_idx1] = new_route1
        solution.truck_routes[route_idx2] = new_route2
        
        return solution
    
    def local_search(self, solution: Solution) -> Solution:
        improved = True
        max_iter = 100
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            # Try all 2-opt moves
            for route_idx in range(len(solution.truck_routes)):
                route = solution.truck_routes[route_idx]
                if len(route) <= 3:
                    continue
                
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Try reversing segment [i, j]
                        new_solution = deepcopy(solution)
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_solution.truck_routes[route_idx] = new_route
                        
                        self.evaluate_solution(new_solution)
                        
                        if new_solution.feasible and new_solution.waiting_time < solution.waiting_time:
                            solution = new_solution
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return solution
    
    def export_solution(self, output_path: str):
        if self.best_solution is None:
            print("No solution to export")
            return
        
        result = {
            'waiting_time': self.best_waiting_time,
            'total_cost': self.best_solution.total_cost,
            'feasible': self.best_solution.feasible,
            'num_trucks_used': len(self.best_solution.truck_routes),
            'routes': []
        }
        
        all_served = set()
        
        for truck_id, route in enumerate(self.best_solution.truck_routes, start=1):
            route_info = {
                'truck_id': truck_id,
                'route': route,
                'customers_served': len(route) - 2
            }
            
            for node in route:
                if node != 0:
                    all_served.add(node)
            current_time = 0.0
            arrival_times = [0.0]
            service_times = []
            
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                
                current_time += self.instance.time_truck[from_node, to_node]
                
                if to_node != 0:
                    node = self.instance.nodes[to_node]
                    service_start = max(current_time, node.time_window[0])
                    arrival_times.append(current_time)
                    service_times.append(service_start)
                    current_time = service_start + node.service_time
                else:
                    arrival_times.append(current_time)
                    service_times.append(None)
            
            route_info['arrival_times'] = arrival_times
            route_info['service_times'] = service_times
            
            result['routes'].append(route_info)
        
        # Add summary of served customers
        all_customers = set(self.instance.customers)
        result['total_customers'] = len(all_customers)
        result['customers_served'] = len(all_served)
        result['all_customers_served'] = (all_served == all_customers)
        
        if all_served != all_customers:
            result['unserved_customers'] = list(all_customers - all_served)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Solution exported to {output_path}")
        print(f"Customers served: {len(all_served)}/{len(all_customers)}")

    def run(self, max_iterations: int = 1000, time_limit: float = 300) -> Solution:
        start_time = time.time()
        print("Generating initial solution")
        current_solution = self.generate_initial_solution()
        self.evaluate_solution(current_solution)
        
        if current_solution.feasible:
            self.best_solution = deepcopy(current_solution)
            self.best_waiting_time = current_solution.waiting_time
            print(f"Initial waiting time: {self.best_waiting_time:.4f}")
        
        iteration = 0
        no_improvement = 0
        
        while iteration < max_iterations and (time.time() - start_time) < time_limit:
            iteration += 1
            new_solution = self.perturb_solution(current_solution)
            self.evaluate_solution(new_solution)
            
            new_solution = self.local_search(new_solution)
            
            if new_solution.feasible:
                if new_solution.waiting_time < current_solution.waiting_time:
                    current_solution = new_solution
                    no_improvement = 0
                    
                    if new_solution.waiting_time < self.best_waiting_time:
                        self.best_solution = deepcopy(new_solution)
                        self.best_waiting_time = new_solution.waiting_time
                        print(f"Iteration {iteration}: New best waiting time = {self.best_waiting_time:.4f}")
                else:
                    no_improvement += 1
            if no_improvement > 50:
                print(f"Iteration {iteration}: Diversifying...")
                current_solution = self.generate_initial_solution()
                self.evaluate_solution(current_solution)
                no_improvement = 0
        
        elapsed = time.time() - start_time
        print(f"\nLKH3 completed in {elapsed:.2f} seconds")
        print(f"Best waiting time: {self.best_waiting_time:.4f}")
        
        return self.best_solution

def main():
    parser = argparse.ArgumentParser(description='LKH3 for CTDTWB')
    parser.add_argument('--filename', type=str, required=True,
                        help='Input JSON filename')
    parser.add_argument('--data_root', type=str, default='../data/generated/data',
                        help='Data root directory')
    parser.add_argument('--output_root', type=str, default='./result_lkh3',
                        help='Output root directory')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Time limit in seconds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Build file path
    parts = args.filename.split('_')
    subfolder = parts[1] if len(parts) > 1 else ''
    data_path = os.path.join(args.data_root, subfolder, args.filename)
    
    if not os.path.exists(data_path):
        # Try direct path
        data_path = args.filename
        if not os.path.exists(data_path):
            print(f"Error: File not found: {data_path}")
            return
    
    print(f"Loading instance from: {data_path}")
    
    # Load instance
    instance = CTDTWB_Instance(data_path)
    print(f"Instance loaded: {instance.num_customers} customers, "
          f"{instance.num_trucks} trucks, {instance.num_drones} drones")
    
    # Run LKH3
    solver = LKH3_CTDTWB(instance, seed=args.seed)
    solution = solver.run(
        max_iterations=args.max_iterations,
        time_limit=args.time_limit
    )
    
    # Export solution
    os.makedirs(args.output_root, exist_ok=True)
    output_filename = args.filename.replace('.json', '_lkh3.json')
    output_path = os.path.join(args.output_root, output_filename)
    solver.export_solution(output_path)
    
    print("\n=== Solution Summary ===")
    print(f"Waiting Time: {solution.waiting_time:.4f} hours")
    print(f"Total Cost: {solution.total_cost:.2f}")
    print(f"Trucks Used: {len(solution.truck_routes)}")
    print(f"Feasible: {solution.feasible}")


if __name__ == '__main__':
    main()