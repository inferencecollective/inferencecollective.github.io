"""Lab 5: Composite 4-panel EDA statistical dashboard with triple export (PNG, SVG, PDF)."""
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
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")

    # Panel 1 (0, 0): Distribution of delivery minutes with KDE
    sns.histplot(data=df, x="delivery_minutes", hue="region", kde=True, bins=16, ax=axs[0, 0])
    axs[0, 0].set_title("A. Delivery Time Distribution (KDE Overlay)", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Delivery Time (minutes)")
    axs[0, 0].set_ylabel("Frequency")

    # Panel 2 (0, 1): Boxplot of revenue by marketing channel
    order = df.groupby("channel")["revenue"].median().sort_values().index
    sns.boxplot(data=df, x="channel", y="revenue", hue="channel", legend=False, order=order, ax=axs[0, 1], palette="Blues")
    axs[0, 1].set_title("B. Order Revenue Spread by Channel", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Acquisition Channel")
    axs[0, 1].set_ylabel("Revenue (USD)")

    # Panel 3 (1, 0): Scatterplot with regression trend: delivery vs satisfaction
    sns.regplot(
        data=df,
        x="delivery_minutes",
        y="satisfaction",
        scatter_kws={"alpha": 0.45, "s": 25, "color": "#2563eb"},
        line_kws={"color": "#dc2626", "linewidth": 2},
        ax=axs[1, 0]
    )
    axs[1, 0].set_title("C. Delivery Time vs Customer Satisfaction", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Delivery Time (minutes)")
    axs[1, 0].set_ylabel("Satisfaction Rating (1-5)")

    # Panel 4 (1, 1): Correlation matrix heatmap
    cols = ["revenue", "delivery_minutes", "satisfaction", "shipping_cost"]
    corr = df[cols].corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        linecolor="white",
        ax=axs[1, 1]
    )
    axs[1, 1].set_title("D. Key Metric Correlation Matrix", fontsize=11, fontweight="bold")

    fig.suptitle("Exploratory Data Analysis (EDA) Statistical Dashboard (Synthetic)", fontsize=14, fontweight="bold")

    png_path = output_dir / "05-seaborn-eda-dashboard.png"
    svg_path = output_dir / "05-seaborn-eda-dashboard.svg"
    pdf_path = output_dir / "05-seaborn-eda-dashboard.pdf"

    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}, {svg_path}, and {pdf_path}")


if __name__ == "__main__":
    generate_charts()
