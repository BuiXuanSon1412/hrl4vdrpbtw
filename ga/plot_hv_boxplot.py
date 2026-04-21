import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# --- Configuration ---
INPUT_CSV = "./processed_hv_data.csv"
OUTPUT_DIR = "./img/hv_boxplots"
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
COLORS = ["#3D5E95", "#458855", "#c44145", "#e66c15"]


def plot_hv_boxplot():
    df = pd.read_csv(INPUT_CSV)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    all_data = []
    all_positions = []
    color_mapping = []

    group_width = 0.8
    offset_step = group_width / len(NUM_NODES)

    for dist_idx, dist in enumerate(DISTRIBUTIONS):
        base_pos = dist_idx + 1
        for size_idx, num_nodes in enumerate(NUM_NODES):
            pos = (
                base_pos
                - (group_width / 2)
                + (size_idx * offset_step)
                + (offset_step / 2)
            )

            # Filter data from CSV
            subset = df[(df["distribution"] == dist) & (df["num_nodes"] == num_nodes)]
            hv_values = subset["hv"].tolist()

            if hv_values:
                all_data.append(hv_values)
                all_positions.append(pos)
                color_mapping.append(COLORS[size_idx])

    # Create Boxplot with updated line widths
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=offset_step * 0.8,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], color_mapping):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # --- Styling from plot_hv_from_csv1.py ---
    ax.set_xlabel("Coordinate distribution", fontsize=30)
    ax.set_ylabel("HV", fontsize=30)

    # Tick parameters (labelsize 24, width 3)
    ax.tick_params(axis="both", labelsize=24, width=3)

    # X-axis formatting
    ax.set_xticks(range(1, len(DISTRIBUTIONS) + 1))
    ax.set_xticklabels(DISTRIBUTIONS)

    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Legend Styling (2 columns, larger font)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            marker="s",
            markerfacecolor=COLORS[i],
            markersize=15,
            label=f"{NUM_NODES[i]} nodes",
        )
        for i in range(len(NUM_NODES))
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,  # Set to 2 columns as requested in your reference script
        fontsize=20,
        frameon=True,
    )

    plt.tight_layout()
    save_path = output_dir / "CIAGEA_HV_grouped_boxplot.pdf"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved grouped plot to: {save_path}")


if __name__ == "__main__":
    plot_hv_boxplot()
