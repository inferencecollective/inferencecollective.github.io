"""Lab 4: Small-multiple campaign trend facets using relplot and FacetGrid."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "campaigns_weekly.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_charts(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    sns.set_theme(style="whitegrid", palette="colorblind")

    # Figure-level relplot manages its own FacetGrid
    g = sns.relplot(
        data=df,
        x="week",
        y="signups",
        hue="channel",
        col="region",
        kind="line",
        errorbar=None,
        height=3.5,
        aspect=1.2,
        marker="o",
        linewidth=2
    )

    g.set_axis_labels("Campaign Week (1–12)", "Weekly Qualified Signups")
    g.set_titles(col_template="Region: {col_name}", size=11, weight="bold")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Weekly Multi-Channel Acquisition Trends by Region (Synthetic)", fontsize=13, fontweight="bold")

    png_path = output_dir / "04-seaborn-faceted-trends.png"
    svg_path = output_dir / "04-seaborn-faceted-trends.svg"
    g.figure.savefig(png_path, dpi=160, bbox_inches="tight")
    g.figure.savefig(svg_path, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_charts()
