"""Lab 3: Feature correlation matrix and annotated heatmap."""
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

    cols = ["revenue", "delivery_minutes", "satisfaction", "shipping_cost"]
    corr = df[cols].corr(numeric_only=True)

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
        ax=ax
    )
    ax.set_title("E-Commerce Metric Correlation Grid (Synthetic)", fontsize=12, fontweight="bold")

    png_path = output_dir / "03-seaborn-correlation.png"
    svg_path = output_dir / "03-seaborn-correlation.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_charts()
