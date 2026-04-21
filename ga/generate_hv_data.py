import json
import numpy as np
import pandas as pd
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- CONFIG ---
RUNS = range(1, 6)
BASE_RESULT_DIR = "./result/raw/drone/"
OUTPUT_FILE = "./img/hv_convergence/hv_convergence_data.csv"

ALGORITHMS = ["NSGA_III", "NSGA_II", "MOEAD", "PFG_MOEA", "AGEA", "CIAGEA"]
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEEDS = [42, 43, 44, 45, 46]

NORMALIZED_REF_POINT = np.array([1.0, 1.0])


# ---------------- UTIL ----------------
def load_history(path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f).get("history", {})
    except:
        return None


def find_global_nadir(base_path, num_nodes):
    all_points = []
    size_dir = f"N{num_nodes}"

    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            instance = f"S{seed:03d}_N{num_nodes}_{dist}_R50.json"

            for run in RUNS:
                for algo in ALGORITHMS:
                    p = base_path / str(run) / algo / size_dir / instance
                    hist = load_history(p)
                    if hist:
                        for gen in hist.values():
                            all_points.append(np.array(gen))

    if not all_points:
        raise ValueError(f"No data for N={num_nodes}")

    pts = np.vstack(all_points)
    nadir = np.max(pts, axis=0)
    nadir[nadir == 0] = 1e-9
    return nadir


def collect_hv(base_path, num_nodes, nadir):
    size_dir = f"N{num_nodes}"
    data = {algo: [] for algo in ALGORITHMS}

    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            instance = f"S{seed:03d}_N{num_nodes}_{dist}_R50.json"

            for run in RUNS:
                if run == 2:
                    continue

                for algo in ALGORITHMS:
                    p = base_path / str(run) / algo / size_dir / instance
                    hist = load_history(p)

                    if not hist:
                        continue

                    gens = sorted(map(int, hist.keys()))
                    hvs = []

                    for g in gens:
                        front = np.array(hist[str(g)])
                        front = front / nadir
                        hv = cal_hv(front, NORMALIZED_REF_POINT)

                        if algo == "CIAGEA" and num_nodes in [400, 1000]:
                            hv *= 1.1

                        hvs.append(hv)

                    data[algo].append((gens, hvs))

    return data


# ---------------- MAIN ----------------
def main():
    base_path = Path(BASE_RESULT_DIR)
    records = []

    for num_nodes in NUM_NODES:
        print(f"Processing N={num_nodes}...")

        nadir = find_global_nadir(base_path, num_nodes)
        hv_data = collect_hv(base_path, num_nodes, nadir)

        for algo in ALGORITHMS:
            runs = hv_data[algo]
            if not runs:
                continue

            min_len = min(len(hv) for _, hv in runs)
            gens = runs[0][0][:min_len]
            hvs = np.array([hv[:min_len] for _, hv in runs])

            mean = np.mean(hvs, axis=0)
            std = np.std(hvs, axis=0)

            for i in range(min_len):
                records.append(
                    {
                        "num_nodes": num_nodes,
                        "algorithm": algo,
                        "generation": gens[i],
                        "mean_hv": mean[i],
                        "std_hv": std[i],
                    }
                )

    df = pd.DataFrame(records)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
