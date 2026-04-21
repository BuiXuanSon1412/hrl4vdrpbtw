import json
import numpy as np
import pandas as pd
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration ---
RUNS = range(1, 6)
BASE_RESULT_DIR = "./result/raw/drone/"
ALGORITHM = "CIAGEA"
NORMALIZED_REF_POINT = np.array([1.0, 1.0])
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEEDS = [42, 43, 44, 45, 46]
OUTPUT_CSV = "./processed_hv_data.csv"


def get_algorithm_history(json_path):
    """Load history from a JSON result file."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("history", {})
    except Exception:
        return None


def find_global_nadir(base_path, num_nodes, distribution):
    """Find global nadir for a specific size and distribution across all seeds/runs."""
    all_points = []
    size_dir = f"N{num_nodes}"
    for seed in SEEDS:
        instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"
        for run in RUNS:
            json_path = base_path / str(run) / ALGORITHM / size_dir / instance_file
            history = get_algorithm_history(json_path)
            if history:
                for gen_data in history.values():
                    all_points.append(np.array(gen_data))
    if not all_points:
        return None
    combined_points = np.vstack(all_points)
    global_nadir = np.max(combined_points, axis=0)
    global_nadir[global_nadir == 0] = 1e-9
    return global_nadir


def main():
    base_path = Path(BASE_RESULT_DIR)
    results = []

    for dist in DISTRIBUTIONS:
        for num_nodes in NUM_NODES:
            print(f"Processing {dist} - N{num_nodes}...")
            # Calculate the nadir point for normalization
            global_nadir = find_global_nadir(base_path, num_nodes, dist)
            if global_nadir is None:
                continue

            size_dir = f"N{num_nodes}"
            for seed in SEEDS:
                # FIXED: correctly using the 'dist' variable here
                instance_file = f"S{seed:03d}_N{num_nodes}_{dist}_R50.json"
                for run in RUNS:
                    json_path = (
                        base_path / str(run) / ALGORITHM / size_dir / instance_file
                    )
                    history = get_algorithm_history(json_path)

                    if history and len(history) > 0:
                        # Get final generation HV
                        final_gen = str(max([int(g) for g in history.keys()]))
                        final_front = np.array(history[final_gen])

                        normalized_front = final_front / global_nadir
                        hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)

                        results.append(
                            {
                                "distribution": dist,
                                "num_nodes": num_nodes,
                                "seed": seed,
                                "run": run,
                                "hv": hv,
                            }
                        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
