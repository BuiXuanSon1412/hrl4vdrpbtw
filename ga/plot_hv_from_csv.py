import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================
# CONFIG
# =========================
INPUT_FILE = "./img/hv_convergence/hv_convergence_data.csv"
OUTPUT_DIR = "./img/hv_convergence/"

ALGORITHMS = ["NSGA_III", "NSGA_II", "MOEAD", "PFG_MOEA", "AGEA", "CIAGEA"]
NUM_NODES = [100, 200, 400, 1000]

SAVE_SEPARATE = True


# =========================
# PLOT FUNCTION
# =========================
def plot_one(ax, df, num_nodes):
    subset = df[df["num_nodes"] == num_nodes]

    for algo in ALGORITHMS:
        data = subset[subset["algorithm"] == algo]
        if data.empty:
            continue

        gens = data["generation"]
        mean = data["mean_hv"]
        std = data["std_hv"]

        ax.plot(
            gens,
            mean,
            linewidth=2,
            label=algo.replace("_", "-"),
        )

        ax.fill_between(
            gens,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # 1. Chỉnh xlabel và ylabel về cỡ 30
    ax.set_xlabel("Generation", fontsize=30)
    ax.set_ylabel("HV", fontsize=30)

    # 2. Chỉnh các số trên trục (xticks, yticks) to lên (ví dụ: cỡ 24)
    ax.tick_params(axis="both", labelsize=24, width=3)

    # 3. Chỉ hiển thị các mốc thế hệ 50, 100, 150, 200 trên trục X
    ax.set_xticks([50, 100, 150, 200])

    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Để legend thành 2 cột và chỉnh lại cỡ chữ cho phù hợp
    ax.legend(fontsize=18, ncol=2, loc="upper left", bbox_to_anchor=(0, 1))


# =========================
# MAIN (Giữ nguyên)
# =========================
def main():
    df = pd.read_csv(INPUT_FILE)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if SAVE_SEPARATE:
        for num_nodes in NUM_NODES:
            fig, ax = plt.subplots(
                figsize=(8, 7)
            )  # Tăng nhẹ figsize để không bị đè chữ

            plot_one(ax, df, num_nodes)

            output_path = Path(OUTPUT_DIR) / f"hv_{num_nodes}_nodes.pdf"

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved plot to {output_path}")

    else:
        output_path = Path(OUTPUT_DIR) / "hv_all_plots.pdf"
        with PdfPages(output_path) as pdf:
            for num_nodes in NUM_NODES:
                fig, ax = plt.subplots(figsize=(8, 7))
                plot_one(ax, df, num_nodes)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Saved multi-page PDF to {output_path}")


if __name__ == "__main__":
    main()
