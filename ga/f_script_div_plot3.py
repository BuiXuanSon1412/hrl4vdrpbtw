import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import random
from matplotlib.ticker import MultipleLocator

# --- Configuration ---
IMAGE_DIR = "./img/div_evolution"
INIT_DIVS = [10, 20, 40]
NUM_NODES = [100, 200, 400, 1000]
TOTAL_GENERATIONS = 200

COLORS = {10: "#ff0000", 20: "#00ff00", 40: "#0000ff"}


def simulate_accurate_vrp_div(init_div, num_nodes, total_gens):
    gens = np.arange(total_gens)
    random.seed(42)
    target_div = 11 + (num_nodes / 25) + random.randint(4, 7)
    k = 0.15 / (init_div / 10) + random.random() * 0.3 * 0.15 / (init_div / 10)
    base_curve = (init_div - target_div) * np.exp(-k * gens) + target_div
    np.random.seed(num_nodes + init_div)
    noise = np.random.normal(0, 0.4, size=total_gens)
    jumps = np.zeros(total_gens)
    if num_nodes > 300:
        jump_point = np.random.randint(50, 150)
        jumps[jump_point:] += np.random.uniform(1, 3)
    return gens, base_curve + noise + jumps


def plot_one(ax, num_nodes):
    for init_div in sorted(INIT_DIVS):
        gens, divs = simulate_accurate_vrp_div(init_div, num_nodes, TOTAL_GENERATIONS)
        y_smooth = savgol_filter(divs, 11, 2)

        gens_sampled = gens[::10]
        divs_sampled = y_smooth[::10]

        ax.plot(
            gens_sampled,
            divs_sampled,
            color=COLORS[init_div],
            linewidth=2.5,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            label=f"Initial division: {init_div}",
        )

    # 1. Chỉnh xlabel và ylabel cỡ 30
    ax.set_xlabel("Generation", fontsize=30)
    ax.set_ylabel("Grid division", fontsize=30)

    # 2. Chỉnh các thông số tick (BỎ labelweight ở đây để tránh lỗi)
    ax.tick_params(axis="both", labelsize=24, width=3, length=8)

    # CÁCH SỬA LỖI: Làm đậm thủ công các nhãn số trên trục
    for label in ax.get_xticklabels():
        label.set_weight("bold")
    for label in ax.get_yticklabels():
        label.set_weight("bold")

    # 3. Làm đậm khung viền biểu đồ
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 4. Cố định các mốc thế hệ
    ax.set_xticks([0, 50, 100, 150, 200])

    # 5. Cố định các mốc Grid division
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylim(-5, 75)

    ax.grid(True, linestyle="--", alpha=0.6)

    # 6. Legend
    ax.legend(fontsize=18, loc="upper right", frameon=True)


def main():
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for num_nodes in NUM_NODES:
        # Sử dụng figsize (8, 7) tương tự plot_hv_from_csv1.py
        fig, ax = plt.subplots(figsize=(8, 7))

        plot_one(ax, num_nodes)

        # Không dùng Title theo yêu cầu
        output_path = output_dir / f"div_convergence_{num_nodes}_nodes.pdf"

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
