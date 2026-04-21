import numpy as np
import random
import json
import argparse
import os


def generate_coords(num_customers, max_coord, dist_type, seed):
    np.random.seed(seed)
    random.seed(seed)

    if dist_type == "R":
        return np.random.uniform(0, max_coord, size=(num_customers, 2)).tolist()

    elif dist_type == "C":
        coords = []
        remaining_nodes = num_customers
        centers = []

        # Define cluster spread (standard deviation)
        if num_customers <= 200:
            std_dev = max_coord / 25  # ~4.0 km
        elif num_customers <= 400:
            std_dev = max_coord / 32  # ~3.125 km
        else:
            std_dev = max_coord / 40  # ~2.5 km
        # Minimum distance between cluster centers (4 * std_dev ensures minimal overlap)
        min_dist = std_dev * 4  # ~16.0 km

        while remaining_nodes > 0:
            current_cluster_size = min(remaining_nodes, random.randint(10, 15))

            proposal = np.random.uniform(5, max_coord - 5, size=(2,))
            # Find a valid center that doesn't overlap existing ones
            valid_center = False
            attempts = 0
            while not valid_center and attempts < 1000:
                # Keep centers away from edges to avoid clipping distortion
                proposal = np.random.uniform(5, max_coord - 5, size=(2,))

                if not centers:
                    valid_center = True
                else:
                    # Check distance against all existing centers
                    dists = [np.linalg.norm(proposal - c) for c in centers]
                    if min(dists) >= min_dist:
                        valid_center = True
                attempts += 1

            centers.append(proposal)

            # Generate nodes for this distinct cluster
            for _ in range(current_cluster_size):
                point = np.random.normal(proposal, std_dev)
                coords.append(np.clip(point, 0, max_coord).tolist())

            remaining_nodes -= current_cluster_size

        return coords

    elif dist_type == "RC":
        n_c = num_customers // 2
        return generate_coords(n_c, max_coord, "C", seed) + generate_coords(
            num_customers - n_c, max_coord, "R", seed + 500
        )


def create_instance(config, n, dist_type, ratio, seed):
    random.seed(seed)
    np.random.seed(seed)

    max_coord = config["MAX_COORD"]
    coords = generate_coords(n, max_coord, dist_type, seed)
    if coords is None:
        raise ValueError(
            "Failed to generate coordinates: generate_coords returned None"
        )
    depot_coord = [max_coord / 2, max_coord / 2]

    # Read time horizon from config, default to 12.0 for backward compatibility
    t_max_system = float(config.get("T_MAX_SYSTEM_H", 24.0))

    num_trucks = config["FLEET_SIZES"].get(str(n), max(1, n // 8))

    nodes = []
    n_linehaul = int(n * ratio)
    types = ["LINEHAUL"] * n_linehaul + ["BACKHAUL"] * (n - n_linehaul)
    random.shuffle(types)

    for i in range(n):
        node_type = types[i]
        demand_range = (
            config["DEMAND_RANGE_LINEHAUL"]
            if node_type == "LINEHAUL"
            else config["DEMAND_RANGE_BACKHAUL"]
        )

        # RESEARCH UPGRADE: Ensure travel feasibility
        dist_km = np.linalg.norm(np.array(coords[i]) - np.array(depot_coord))
        min_reach_time = dist_km / config["V_TRUCK_KM_H"]

        # Earliest start is travel time + buffer
        ready_h = random.uniform(min_reach_time * 1.1, t_max_system * 0.7)

        # Scaled window width based on distance (farther nodes get wider windows)
        width_h = random.uniform(
            1.0, 1.0 + (config["TIME_WINDOW_SCALING_FACTOR"] * (dist_km / max_coord))
        )

        nodes.append(
            {
                "id": i + 1,
                "coord": coords[i],
                "demand": random.randint(demand_range[0], demand_range[1]),
                "type": node_type,
                "tw_h": [
                    round(ready_h, 4),
                    round(min(ready_h + width_h, t_max_system), 4),
                ],
            }
        )

    # Return structure identical to S046_N5_Z_3G_R50.json
    return {
        "Config": {
            "General": {
                "NUM_CUSTOMERS": n,
                "NUM_NODES": n + 1,
                "MAX_COORD_KM": max_coord,
                "T_MAX_SYSTEM_H": t_max_system,
                "TIME_WINDOW_SCALING_FACTOR": config["TIME_WINDOW_SCALING_FACTOR"],
                "COORD_DISTRIBUTION": dist_type,
                "NUM_CLUSTERS": max(1, n // 15) if dist_type != "R" else 0,
            },
            "Vehicles": {
                "NUM_TRUCKS": num_trucks,
                "NUM_DRONES": num_trucks,
                "V_TRUCK_KM_H": config["V_TRUCK_KM_H"],
                "V_DRONE_KM_H": config["V_DRONE_KM_H"],
                "CAPACITY_TRUCK": config["CAPACITY_TRUCK"],
                "CAPACITY_DRONE": config["CAPACITY_DRONE"],
                "DRONE_TAKEOFF_MIN": config["DRONE_TAKEOFF_MIN"],
                "DRONE_LANDING_MIN": config["DRONE_LANDING_MIN"],
                "SERVICE_TIME_MIN": config["SERVICE_TIME_MIN"],
                "DRONE_DURATION_H": config["DRONE_DURATION_H"],
            },
            "Depot": {
                "id": 0,
                "coord": depot_coord,
                "time_window_h": [0.0, t_max_system],
            },
        },
        "Nodes": nodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to reflect requested batch range
    parser.add_argument(
        "--n", type=int, choices=[10, 20, 50, 100, 200, 400, 1000], default=400
    )
    parser.add_argument("--dist", choices=["R", "C", "RC"], default="RC")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    output_dir = os.path.join("data", f"N{args.n}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.count):
        current_seed = args.seed + i
        instance_data = create_instance(
            config, args.n, args.dist, args.ratio, current_seed
        )
        filename = (
            f"S{current_seed:03d}_N{args.n}_{args.dist}_R{int(args.ratio * 100)}.json"
        )
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(instance_data, f, indent=4)
        print(f"Generated: {filename}")
