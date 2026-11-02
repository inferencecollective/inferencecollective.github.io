"""Lab 2: Categorical comparisons using box, strip, and violin plots."""
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    # Panel 1: Box plot overlaid with jittered strip plot
    order = df.groupby("channel")["revenue"].median().sort_values().index
    sns.boxplot(data=df, x="channel", y="revenue", order=order, ax=axs[0], color="#93c5fd", boxprops=dict(alpha=0.7))
    sns.stripplot(data=df, x="channel", y="revenue", order=order, color="#1e3a8a", alpha=0.35, jitter=0.2, size=3, ax=axs[0])
    axs[0].set_title("Revenue by Channel: Box + Strip", fontsize=11, fontweight="bold")
    axs[0].set_xlabel("Acquisition Channel")
    axs[0].set_ylabel("Revenue (USD)")

    # Panel 2: Violin plot showing distribution density shape
    sns.violinplot(data=df, x="region", y="delivery_minutes", hue="region", legend=False, cut=0, ax=axs[1], palette="Set2")
    axs[1].set_title("Delivery Time by Region: Violin Density", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Region")
    axs[1].set_ylabel("Delivery Time (minutes)")

    fig.suptitle("Categorical Variation and Density Comparisons (Synthetic)", fontsize=13, fontweight="bold")

    png_path = output_dir / "02-seaborn-categorical.png"
    svg_path = output_dir / "02-seaborn-categorical.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_charts()
