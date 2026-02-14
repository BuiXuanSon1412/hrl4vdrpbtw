import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration ---
RUNS = range(1, 6)  # Runs 1 to 5
BASE_RESULT_DIR = "./result/test/1"
IMAGE_DIR = "./img/test/hv_convergence"
ALGORITHMS = ["CIAGEA", "PFG_MOEA"]
NORMALIZED_REF_POINT = np.array([1.0, 1.0])

# Problem configurations
NUM_NODES = [100]
DISTRIBUTIONS = ["C"]
SEEDS = [42]


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


def find_global_nadir_for_batch(base_path, num_nodes, distribution):
    """
    Find global nadir point for a batch across all instances, runs, and algorithms.

    Args:
        base_path: Base result directory path
        num_nodes: Number of nodes (100, 200, 400, 1000)
        distribution: Distribution type (C, R, RC)

    Returns:
        Global nadir point as numpy array
    """
    all_points = []
    size_dir = f"N{num_nodes}"

    for seed in SEEDS:
        instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

        for run in RUNS:
            for algo in ALGORITHMS:
                json_path = base_path / str(run) / algo / size_dir / instance_file
                history = get_algorithm_history(json_path)

                if history:
                    for gen_data in history.values():
                        all_points.append(np.array(gen_data))

    if not all_points:
        raise ValueError(f"No data found for batch N{num_nodes}_{distribution}")

    combined_points = np.vstack(all_points)
    global_nadir = np.max(combined_points, axis=0)
    global_nadir[global_nadir == 0] = 1e-9

    return global_nadir


def collect_hv_data_for_batch(base_path, num_nodes, distribution, global_nadir):
    """
    Collect HV convergence data for all algorithms in a batch.

    Args:
        base_path: Base result directory path
        num_nodes: Number of nodes
        distribution: Distribution type
        global_nadir: Global nadir point for normalization

    Returns:
        Dictionary mapping algorithm name to list of HV arrays (one per instance×run)
    """
    size_dir = f"N{num_nodes}"
    algo_hv_data = {algo: [] for algo in ALGORITHMS}

    for seed in SEEDS:
        instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

        for run in RUNS:
            for algo in ALGORITHMS:
                json_path = base_path / str(run) / algo / size_dir / instance_file
                history = get_algorithm_history(json_path)

                if history and len(history) > 0:
                    # Extract and sort generations
                    gens = sorted([int(g) for g in history.keys()])

                    # Calculate HV for each generation
                    hv_values = []
                    for g in gens:
                        front = np.array(history[str(g)])
                        normalized_front = front / global_nadir
                        hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)
                        hv_values.append(hv)

                    algo_hv_data[algo].append((gens, hv_values))

    return algo_hv_data


def plot_batch_convergence(num_nodes, distribution):
    """
    Plot average HV convergence for a batch.

    Args:
        num_nodes: Number of nodes
        distribution: Distribution type
    """
    base_path = Path(BASE_RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_name = f"N{num_nodes}_{distribution}"
    print(f"\nProcessing batch: {batch_name}")

    try:
        # 1. Find global nadir for this batch
        global_nadir = find_global_nadir_for_batch(base_path, num_nodes, distribution)
        print(f"  Global nadir: {global_nadir}")

        # 2. Collect HV data for all algorithms
        algo_hv_data = collect_hv_data_for_batch(
            base_path, num_nodes, distribution, global_nadir
        )

        # 3. Plot
        plt.figure(figsize=(10, 6))

        for algo in ALGORITHMS:
            hv_runs = algo_hv_data[algo]

            if not hv_runs:
                print(f"  Warning: No data for {algo}")
                continue

            # Find minimum length across all runs (for alignment)
            min_len = min(len(hv_vals) for _, hv_vals in hv_runs)

            # Align all runs to same length and stack
            aligned_gens = hv_runs[0][0][:min_len]  # Use first run's generations
            aligned_hvs = np.array([hv_vals[:min_len] for _, hv_vals in hv_runs])

            # Calculate mean and std
            mean_hv = np.mean(aligned_hvs, axis=0)
            std_hv = np.std(aligned_hvs, axis=0)

            # Plot mean line with shaded std area
            (line,) = plt.plot(aligned_gens, mean_hv, label=algo, linewidth=2)
            plt.fill_between(
                aligned_gens,
                mean_hv - std_hv,
                mean_hv + std_hv,
                color=line.get_color(),
                alpha=0.15,
            )

            print(
                f"  {algo}: {len(hv_runs)} instance×run combinations, "
                f"final HV = {mean_hv[-1]:.4f} ± {std_hv[-1]:.4f}"
            )

        # 4. Formatting
        # plt.title(
        #    f"HV Convergence: {batch_name} (avg over {len(SEEDS)} instances × {len(RUNS)} runs)"
        # )
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # 5. Save
        save_path = output_dir / f"{batch_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved to: {save_path}")

    except Exception as e:
        print(f"  ERROR processing {batch_name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Plot HV convergence for all batches."""
    print("=" * 80)
    print("Generating HV Convergence Plots by Batch")
    print("=" * 80)

    total_batches = len(NUM_NODES) * len(DISTRIBUTIONS)
    current = 0

    for num_nodes in NUM_NODES:
        for dist in DISTRIBUTIONS:
            current += 1
            print(f"\n[{current}/{total_batches}] Processing batch...")
            plot_batch_convergence(num_nodes, dist)

    print("\n" + "=" * 80)
    print(f"Completed! All plots saved to {IMAGE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
