"""Lab 1: Distribution analysis using histogram and ECDF."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_charts(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # 1. Histogram with binned counts
    sns.histplot(data=df, x="delivery_minutes", hue="region", bins=18, multiple="stack", ax=axs[0])
    axs[0].set_title("Histogram: Delivery Time Distribution", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Delivery Time (minutes)")
    axs[0].set_ylabel("Order Count")

    # 2. Empirical Cumulative Distribution Function
    sns.ecdfplot(data=df, x="delivery_minutes", hue="region", ax=axs[1])
    axs[1].set_title("ECDF: Cumulative Proportion Below Threshold", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Delivery Time (minutes)")
    axs[1].set_ylabel("Cumulative Fraction (0.0 - 1.0)")

    fig.suptitle("Delivery Time Analysis: Histogram vs ECDF (Synthetic)", fontsize=13, fontweight="bold")

    png_path = output_dir / "01-seaborn-distributions.png"
    svg_path = output_dir / "01-seaborn-distributions.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_charts()
