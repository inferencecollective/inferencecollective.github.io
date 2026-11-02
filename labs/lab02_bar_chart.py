"""Lab 2: Acquisition channel comparison with horizontal bars and direct value labels."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "channels.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_chart(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    # Sort ascending so highest appears at the top in horizontal bars
    df = df.sort_values("leads", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bars = ax.barh(df["channel"], df["leads"], color="#3b82f6", edgecolor="#1d4ed8")
    ax.bar_label(bars, fmt="%d", padding=5, fontweight="semibold")

    ax.set_xlim(left=0, right=df["leads"].max() * 1.15)
    ax.set_title("Acquisition Channel Performance: Total Leads (Synthetic)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Qualified Leads")
    ax.grid(axis="x", alpha=0.25)

    png_path = output_dir / "02-channel-comparison.png"
    svg_path = output_dir / "02-channel-comparison.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_chart()
