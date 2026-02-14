import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

# --- Configuration ---
RESULT_DIR = "./result/test/"
IMAGE_DIR = "./img/test/detailed_evolution"
ALGORITHM = "AGEA"  # or "CIAGEA"


def load_detailed_history(json_path):
    """Load detailed history from JSON file."""
    detailed_path = json_path.parent / f"{json_path.stem}_detailed.json"

    if not detailed_path.exists():
        print(f"Detailed history not found: {detailed_path}")
        return None

    try:
        with open(detailed_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {detailed_path}: {e}")
        return None


def plot_fitness_evolution_with_grid(detailed_history, output_path):
    """
    Create a comprehensive visualization showing:
    1. All solutions fitness over generations
    2. Pareto front evolution
    3. Grid division evolution
    4. Grid diversity metrics
    """

    gens_data = detailed_history["generations"]
    gens = sorted([int(g) for g in gens_data.keys()])

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # =============================================
    # Plot 1: Fitness Space Evolution (Large plot)
    # =============================================
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(gens)))

    for idx, gen in enumerate(gens):
        gen_data = gens_data[str(gen)]

        # Plot all solutions (faded)
        all_solutions = np.array(gen_data["all_solutions"])
        if len(all_solutions) > 0:
            ax1.scatter(
                all_solutions[:, 0],
                all_solutions[:, 1],
                c=[colors[idx]],
                alpha=0.2,
                s=30,
                edgecolors="none",
            )

        # Plot Pareto front (bold)
        pareto_front = np.array(gen_data["pareto_front"])
        if len(pareto_front) > 0:
            ax1.scatter(
                pareto_front[:, 0],
                pareto_front[:, 1],
                c=[colors[idx]],
                alpha=0.8,
                s=80,
                edgecolors="black",
                linewidth=1.5,
                label=f"Gen {gen}" if idx % 2 == 0 else "",
                marker="*",
            )

    ax1.set_xlabel("Tardiness", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cost", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Fitness Space Evolution\n(Stars = Pareto Front, Dots = All Solutions)",
        fontsize=16,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=10, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # =============================================
    # Plot 2: Grid Division Evolution
    # =============================================
    ax2 = fig.add_subplot(gs[0, 2])

    div_values = [gens_data[str(g)]["grid_info"]["div"] for g in gens]

    ax2.plot(gens, div_values, "o-", linewidth=2, markersize=8, color="#2ecc71")
    ax2.fill_between(gens, div_values, alpha=0.3, color="#2ecc71")
    ax2.set_xlabel("Generation", fontsize=12)
    ax2.set_ylabel("Grid Divisions", fontsize=12)
    ax2.set_title("Grid Division (div) Evolution", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # =============================================
    # Plot 3: Grid Diversity Metrics
    # =============================================
    ax3 = fig.add_subplot(gs[1, 2])

    unique_grids = [gens_data[str(g)]["grid_info"]["unique_grids"] for g in gens]
    grid_diversity = [gens_data[str(g)]["grid_info"]["grid_diversity"] for g in gens]

    ax3_twin = ax3.twinx()

    line1 = ax3.plot(
        gens,
        unique_grids,
        "o-",
        linewidth=2,
        markersize=6,
        color="#3498db",
        label="Unique Grids",
    )
    line2 = ax3_twin.plot(
        gens,
        grid_diversity,
        "s-",
        linewidth=2,
        markersize=6,
        color="#e74c3c",
        label="Diversity Ratio",
    )

    ax3.set_xlabel("Generation", fontsize=12)
    ax3.set_ylabel("Unique Grid Cells", fontsize=12, color="#3498db")
    ax3_twin.set_ylabel("Diversity Ratio", fontsize=12, color="#e74c3c")
    ax3.set_title("Grid Diversity Metrics", fontsize=14, fontweight="bold")

    ax3.tick_params(axis="y", labelcolor="#3498db")
    ax3_twin.tick_params(axis="y", labelcolor="#e74c3c")

    # Combined legend
    lines = line1 + line2
    labels = [str(line.get_label()) for line in lines]
    ax3.legend(lines, labels, loc="best", fontsize=10)

    ax3.grid(True, linestyle="--", alpha=0.3)

    # =============================================
    # Plot 4: Population Size Evolution
    # =============================================
    ax4 = fig.add_subplot(gs[2, 0])

    pop_sizes = [gens_data[str(g)]["population_stats"]["size"] for g in gens]
    pareto_sizes = [
        gens_data[str(g)]["population_stats"]["pareto_front_size"] for g in gens
    ]

    ax4.plot(
        gens,
        pop_sizes,
        "o-",
        linewidth=2,
        markersize=6,
        color="#9b59b6",
        label="Population Size",
    )
    ax4.plot(
        gens,
        pareto_sizes,
        "s-",
        linewidth=2,
        markersize=6,
        color="#f39c12",
        label="Pareto Front Size",
    )

    ax4.set_xlabel("Generation", fontsize=12)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.set_title("Population Statistics", fontsize=14, fontweight="bold")
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.3)

    # =============================================
    # Plot 5: Ideal and Nadir Point Evolution
    # =============================================
    ax5 = fig.add_subplot(gs[2, 1])

    ideal_obj1 = []
    ideal_obj2 = []
    nadir_obj1 = []
    nadir_obj2 = []

    for gen in gens:
        grid_info = gens_data[str(gen)]["grid_info"]
        if grid_info["ideal_point"]:
            ideal_obj1.append(grid_info["ideal_point"][0])
            ideal_obj2.append(grid_info["ideal_point"][1])
        if grid_info["grid_nadir"]:
            nadir_obj1.append(grid_info["grid_nadir"][0])
            nadir_obj2.append(grid_info["grid_nadir"][1])

    ax5.plot(
        gens[: len(ideal_obj1)],
        ideal_obj1,
        "o-",
        linewidth=2,
        color="#27ae60",
        label="Ideal - Obj1",
    )
    ax5.plot(
        gens[: len(ideal_obj2)],
        ideal_obj2,
        "s-",
        linewidth=2,
        color="#2980b9",
        label="Ideal - Obj2",
    )
    ax5.plot(
        gens[: len(nadir_obj1)],
        nadir_obj1,
        "^--",
        linewidth=2,
        color="#c0392b",
        label="Nadir - Obj1",
        alpha=0.7,
    )
    ax5.plot(
        gens[: len(nadir_obj2)],
        nadir_obj2,
        "v--",
        linewidth=2,
        color="#8e44ad",
        label="Nadir - Obj2",
        alpha=0.7,
    )

    ax5.set_xlabel("Generation", fontsize=12)
    ax5.set_ylabel("Objective Value", fontsize=12)
    ax5.set_title("Ideal & Nadir Point Evolution", fontsize=14, fontweight="bold")
    ax5.legend(loc="best", fontsize=9, ncol=2)
    ax5.grid(True, linestyle="--", alpha=0.3)

    # =============================================
    # Plot 6: Grid Distribution Heatmap (Last Generation)
    # =============================================
    ax6 = fig.add_subplot(gs[2, 2])

    last_gen = str(gens[-1])
    grid_dist = gens_data[last_gen]["grid_info"]["grid_distribution"]

    # Parse grid indices
    grid_coords = []
    grid_counts = []
    for grid_key, count in grid_dist.items():
        coords = eval(grid_key)  # Convert "(x, y)" string to tuple
        grid_coords.append(coords)
        grid_counts.append(count)

    if grid_coords:
        grid_coords = np.array(grid_coords)

        # Create scatter with size based on count
        scatter = ax6.scatter(
            grid_coords[:, 0],
            grid_coords[:, 1],
            s=[c * 50 for c in grid_counts],
            c=grid_counts,
            cmap="YlOrRd",
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
        )

        plt.colorbar(scatter, ax=ax6, label="Solutions per Grid")

        ax6.set_xlabel("Grid Index - Obj1", fontsize=12)
        ax6.set_ylabel("Grid Index - Obj2", fontsize=12)
        ax6.set_title(
            f"Grid Distribution (Gen {gens[-1]})", fontsize=14, fontweight="bold"
        )
        ax6.grid(True, linestyle="--", alpha=0.3)

    # Main title
    config = detailed_history["config"]
    fig.suptitle(
        f"{ALGORITHM} Detailed Evolution Analysis\n"
        f"PopSize={config['pop_size']}, InitDiv={config['init_div']}, "
        f"CR={config['crossover_rate']}, MR={config['mutation_rate']}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comprehensive plot: {output_path}")


def plot_grid_evolution_animation_style(detailed_history, output_path):
    """
    Create a focused plot showing how grid boundaries evolve.
    """
    gens_data = detailed_history["generations"]
    gens = sorted([int(g) for g in gens_data.keys()])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Select 6 generations to display
    display_gens = [gens[i] for i in np.linspace(0, len(gens) - 1, 6, dtype=int)]

    for idx, gen in enumerate(display_gens):
        ax = axes[idx]
        gen_data = gens_data[str(gen)]

        # Plot all solutions
        all_solutions = np.array(gen_data["all_solutions"])
        if len(all_solutions) > 0:
            ax.scatter(
                all_solutions[:, 0],
                all_solutions[:, 1],
                c="lightblue",
                alpha=0.4,
                s=30,
                edgecolors="gray",
                linewidth=0.5,
            )

        # Plot Pareto front
        pareto_front = np.array(gen_data["pareto_front"])
        if len(pareto_front) > 0:
            ax.scatter(
                pareto_front[:, 0],
                pareto_front[:, 1],
                c="red",
                alpha=0.8,
                s=100,
                edgecolors="black",
                linewidth=2,
                marker="*",
                zorder=5,
            )

        # Draw grid lines
        grid_info = gen_data["grid_info"]
        if grid_info["ideal_point"] and grid_info["grid_nadir"]:
            ideal = np.array(grid_info["ideal_point"])
            nadir = np.array(grid_info["grid_nadir"])
            spacing = np.array(grid_info["grid_spacing"])
            div = grid_info["div"]

            # Vertical lines
            for i in range(div + 1):
                x = ideal[0] + i * spacing[0]
                ax.axvline(x, color="green", linestyle="--", alpha=0.3, linewidth=1)

            # Horizontal lines
            for i in range(div + 1):
                y = ideal[1] + i * spacing[1]
                ax.axhline(y, color="green", linestyle="--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Tardiness", fontsize=11)
        ax.set_ylabel("Cost", fontsize=11)
        ax.set_title(
            f"Gen {gen} | div={grid_info['div']} | "
            f"unique_grids={grid_info['unique_grids']}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.2)

    fig.suptitle(
        f"{ALGORITHM} - Grid Evolution Over Time", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved grid evolution plot: {output_path}")


def main():
    """Generate all detailed visualizations."""

    base_path = Path(RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: Visualize a specific instance
    run = 1
    size_dir = "N100"
    instance_file = "S042_N100_RC_R50.json"

    json_path = base_path / str(run) / ALGORITHM / size_dir / instance_file

    print(f"Loading detailed history from: {json_path}")
    detailed_history = load_detailed_history(json_path)

    if not detailed_history:
        print("No detailed history found!")
        return

    instance_name = instance_file.replace(".json", "")

    # Generate visualizations
    print("\n1. Creating comprehensive evolution plot...")
    plot_fitness_evolution_with_grid(
        detailed_history, output_dir / f"{ALGORITHM}_{instance_name}_comprehensive.png"
    )

    print("\n2. Creating grid evolution plot...")
    plot_grid_evolution_animation_style(
        detailed_history, output_dir / f"{ALGORITHM}_{instance_name}_grid_evolution.png"
    )

    print("\n" + "=" * 80)
    print(f"All visualizations saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
