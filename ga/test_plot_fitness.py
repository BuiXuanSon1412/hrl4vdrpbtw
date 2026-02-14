import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import Normalize

# --- Configuration ---
RESULT_DIR = "./result/test/"
IMAGE_DIR = "./img/test/fitness_evolution"
ALGORITHMS = ["NSGA_II", "NSGA_III", "MOEAD", "PFG_MOEA", "AGEA", "CIAGEA"]
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEEDS = [42, 43, 44, 45, 46]
RUN = 1  # Which run to visualize

# Plot every N generations
GENERATION_STEP = 10


def load_history(json_path):
    """Load history from a result JSON file."""
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("history", {})
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def plot_fitness_evolution_single_algorithm(algorithm, num_nodes, distribution, seed):
    """
    Plot fitness evolution for a single algorithm showing points every 10 generations.

    Creates a scatter plot where each generation's Pareto front is shown in different colors.
    """
    base_path = Path(RESULT_DIR)
    output_dir = Path(IMAGE_DIR) / f"N{num_nodes}" / distribution
    output_dir.mkdir(parents=True, exist_ok=True)

    size_dir = f"N{num_nodes}"
    instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

    json_path = base_path / str(RUN) / algorithm / size_dir / instance_file

    history = load_history(json_path)
    if not history:
        print(f"No data for {algorithm} - {instance_file}")
        return

    # Get all generations and filter by step
    all_gens = sorted([int(g) for g in history.keys()])
    selected_gens = [g for g in all_gens if g % GENERATION_STEP == 0]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for generations
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(selected_gens)))
    for idx, gen in enumerate(selected_gens):
        front = np.array(history[str(gen)])

        ax.scatter(
            front[:, 0],  # Objective 1 (tardiness)
            front[:, 1],  # Objective 2 (cost)
            c=[colors[idx]],
            label=f"Gen {gen}",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Tardiness", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title(
        f"{algorithm} - {instance_file}\nFitness Evolution (every {GENERATION_STEP} generations)",
        fontsize=14,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    save_path = (
        output_dir
        / f"{algorithm}_{instance_file.replace('.json', '')}_fitness_evolution.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def plot_fitness_evolution_comparison(num_nodes, distribution, seed):
    """
    Compare multiple algorithms in subplots, showing fitness evolution.
    """
    base_path = Path(RESULT_DIR)
    output_dir = Path(IMAGE_DIR) / "comparison" / f"N{num_nodes}" / distribution
    output_dir.mkdir(parents=True, exist_ok=True)

    size_dir = f"N{num_nodes}"
    instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

    # Create subplots
    n_algos = len(ALGORITHMS)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_algos > 1 else [axes]

    selected_gens = []
    for idx, algorithm in enumerate(ALGORITHMS):
        ax = axes[idx]

        json_path = base_path / str(RUN) / algorithm / size_dir / instance_file
        history = load_history(json_path)

        if not history:
            ax.text(
                0.5,
                0.5,
                f"No data\n{algorithm}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Get selected generations
        all_gens = sorted([int(g) for g in history.keys()])
        selected_gens = [g for g in all_gens if g % GENERATION_STEP == 0]

        # Color map
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(selected_gens)))

        for gen_idx, gen in enumerate(selected_gens):
            front = np.array(history[str(gen)])

            ax.scatter(
                front[:, 0],
                front[:, 1],
                c=[colors[gen_idx]],
                alpha=0.5,
                s=30,
                edgecolors="black",
                linewidth=0.3,
            )

        ax.set_xlabel("Tardiness", fontsize=10)
        ax.set_ylabel("Cost", fontsize=10)
        ax.set_title(algorithm, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide extra subplots
    for idx in range(n_algos, len(axes)):
        axes[idx].axis("off")

    if not selected_gens:
        # Fallback logic
        vmax_val = 1
    else:
        vmax_val = max(selected_gens)

    # Add colorbar legend for generations
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"),
        norm=Normalize(vmin=0, vmax=vmax_val),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_label("Generation", fontsize=12)

    fig.suptitle(
        f"Fitness Evolution Comparison - {instance_file}\n(every {GENERATION_STEP} generations)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    save_path = (
        output_dir
        / f"comparison_{instance_file.replace('.json', '')}_fitness_evolution.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison: {save_path}")


def plot_fitness_animation_style(algorithm, num_nodes, distribution, seed):
    """
    Create a single plot showing the progression with arrows or connecting lines.
    """
    base_path = Path(RESULT_DIR)
    output_dir = Path(IMAGE_DIR) / "animation_style" / f"N{num_nodes}" / distribution
    output_dir.mkdir(parents=True, exist_ok=True)

    size_dir = f"N{num_nodes}"
    instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

    json_path = base_path / str(RUN) / algorithm / size_dir / instance_file
    history = load_history(json_path)

    if not history:
        print(f"No data for {algorithm} - {instance_file}")
        return

    all_gens = sorted([int(g) for g in history.keys()])
    selected_gens = [g for g in all_gens if g % GENERATION_STEP == 0]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot all generations with increasing opacity
    for idx, gen in enumerate(selected_gens):
        front = np.array(history[str(gen)])

        # Alpha increases with generation
        alpha = 0.2 + (0.6 * idx / len(selected_gens))
        size = 20 + (80 * idx / len(selected_gens))

        ax.scatter(
            front[:, 0],
            front[:, 1],
            alpha=alpha,
            s=size,
            c=[plt.get_cmap("plasma")(idx / len(selected_gens))],
            edgecolors="black",
            linewidth=0.5,
            label=f"Gen {gen}" if idx % 2 == 0 else "",  # Label every other
        )

    # Highlight first and last generation
    first_front = np.array(history[str(selected_gens[0])])
    last_front = np.array(history[str(selected_gens[-1])])

    ax.scatter(
        first_front[:, 0],
        first_front[:, 1],
        s=100,
        c="green",
        marker="s",
        label=f"Gen 0 (Start)",
        zorder=5,
        edgecolors="black",
        linewidth=2,
    )

    ax.scatter(
        last_front[:, 0],
        last_front[:, 1],
        s=100,
        c="red",
        marker="*",
        label=f"Gen {selected_gens[-1]} (End)",
        zorder=5,
        edgecolors="black",
        linewidth=2,
    )

    ax.set_xlabel("Tardiness", fontsize=14)
    ax.set_ylabel("Cost", fontsize=14)
    ax.set_title(f"{algorithm} - Fitness Evolution\n{instance_file}", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    save_path = (
        output_dir / f"{algorithm}_{instance_file.replace('.json', '')}_progression.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved animation-style: {save_path}")


def main():
    """Main execution with multiple visualization options."""

    print("=" * 80)
    print("Generating Fitness Evolution Plots")
    print("=" * 80)

    # Example 1: Single algorithm, single instance
    print("\n1. Creating individual algorithm plots...")
    for algorithm in ALGORITHMS:
        plot_fitness_evolution_single_algorithm(
            algorithm=algorithm, num_nodes=100, distribution="RC", seed=42
        )

    # Example 2: Compare all algorithms for one instance
    """ 
    print("\n2. Creating comparison plots...")
    plot_fitness_evolution_comparison(num_nodes=100, distribution="RC", seed=42)
    """
    # Example 3: Animation-style progression
    print("\n3. Creating animation-style plots...")
    for algorithm in ["AGEA"]:
        plot_fitness_animation_style(
            algorithm=algorithm, num_nodes=100, distribution="RC", seed=42
        )

    print("\n" + "=" * 80)
    print(f"All plots saved to {IMAGE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
