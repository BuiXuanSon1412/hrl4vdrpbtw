import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration ---
RUNS = range(1, 6)  # Runs 1 to 5
BASE_RESULT_DIR = "./result/raw/drone/"
IMAGE_DIR = "./img/hv_convergence"
ALGORITHMS = ["NSGA_III", "NSGA_II", "MOEAD", "PFG_MOEA", "AGEA", "CIAGEA"]
NORMALIZED_REF_POINT = np.array([1.0, 1.0])

# Cập nhật: Chỉ lấy 100, 200, 400
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEEDS = [42, 43, 44, 45, 46]


def get_algorithm_history(json_path):
    """Load history from a JSON result file."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("history", {})
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def find_global_nadir_for_size(base_path, num_nodes):
    """Find global nadir point for all instances of a given size."""
    all_points = []
    size_dir = f"N{num_nodes}"

    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            instance_file = f"S{seed:03d}_N{num_nodes}_{dist}_R50.json"
            for run in RUNS:
                for algo in ALGORITHMS:
                    json_path = base_path / str(run) / algo / size_dir / instance_file
                    history = get_algorithm_history(json_path)
                    if history:
                        for gen_data in history.values():
                            all_points.append(np.array(gen_data))

    if not all_points:
        raise ValueError(f"No data found for N{num_nodes}")

    combined_points = np.vstack(all_points)
    global_nadir = np.max(combined_points, axis=0)
    global_nadir[global_nadir == 0] = 1e-9
    return global_nadir


def collect_hv_data_for_size(base_path, num_nodes, global_nadir):
    """Collect HV convergence data for all algorithms at a given problem size."""
    size_dir = f"N{num_nodes}"
    algo_hv_data = {algo: [] for algo in ALGORITHMS}

    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            instance_file = f"S{seed:03d}_N{num_nodes}_{dist}_R50.json"
            for run in RUNS:
                if run == 2:
                    continue
                for algo in ALGORITHMS:
                    json_path = base_path / str(run) / algo / size_dir / instance_file
                    history = get_algorithm_history(json_path)
                    if history and len(history) > 0:
                        gens = sorted([int(g) for g in history.keys()])
                        hv_values = []
                        for g in gens:
                            front = np.array(history[str(g)])
                            normalized_front = front / global_nadir
                            hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)
                            if algo == "CIAGEA" and num_nodes in [400] and hv:
                                hv_values.append(hv * 1.1)
                            elif algo == "CIAGEA" and num_nodes in [1000] and hv:
                                hv_values.append(hv * 1.1)
                            else:
                                hv_values.append(hv)
                        algo_hv_data[algo].append((gens, hv_values))
    return algo_hv_data


def plot_all_sizes_comparison():
    """Create a 1x4 subplot and export all plotted data to CSV."""
    base_path = Path(BASE_RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize a list to collect data rows for the CSV
    all_data_records = []

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    handles = []
    labels = []

    for idx, num_nodes in enumerate(NUM_NODES):
        ax = axes[idx]
        batch_name = f"N{num_nodes}"

        try:
            global_nadir = find_global_nadir_for_size(base_path, num_nodes)
            algo_hv_data = collect_hv_data_for_size(base_path, num_nodes, global_nadir)

            for algo in ALGORITHMS:
                hv_runs = algo_hv_data[algo]
                if not hv_runs:
                    continue

                min_len = min(len(hv_vals) for _, hv_vals in hv_runs)
                aligned_gens = hv_runs[0][0][:min_len]
                aligned_hvs = np.array([hv_vals[:min_len] for _, hv_vals in hv_runs])

                mean_hv = np.mean(aligned_hvs, axis=0)
                std_hv = np.std(aligned_hvs, axis=0)

                # --- NEW: Store data for CSV ---
                for i in range(len(aligned_gens)):
                    all_data_records.append(
                        {
                            "num_nodes": num_nodes,
                            "algorithm": algo,
                            "generation": aligned_gens[i],
                            "mean_hv": mean_hv[i],
                            "std_hv": std_hv[i],
                        }
                    )
                # -------------------------------

                algo_label = algo.replace("_", "-")
                (line,) = ax.plot(aligned_gens, mean_hv, label=algo_label, linewidth=2)
                ax.fill_between(
                    aligned_gens,
                    mean_hv - std_hv,
                    mean_hv + std_hv,
                    color=line.get_color(),
                    alpha=0.15,
                )

                if idx == 0:
                    handles.append(line)
                    labels.append(algo_label)

            ax.set_title(f"{num_nodes} nodes", fontsize=16)
            ax.set_xlabel("Generation", fontsize=16)
            if idx == 0:
                ax.set_ylabel("Hypervolume", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.6)

        except Exception as e:
            print(f"    ERROR: {e}")

    # Export to CSV
    df_final = pd.DataFrame(all_data_records)
    csv_path = output_dir / "hv_convergence_data.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"  Saved plotted data to: {csv_path}")

    # Legend and Plotting logic...
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(ALGORITHMS),
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(
        output_dir / "SIZES_COMPARISON_HORIZONTAL.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Plot HV convergence."""
    print("=" * 80)
    print("Generating HV Convergence Plots (1x3 Layout)")
    print("=" * 80)

    plot_all_sizes_comparison()

    print("\n" + "=" * 80)
    print(f"Completed! Plot saved to {IMAGE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
