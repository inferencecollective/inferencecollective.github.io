"""Lab 5: Activity heatmap with cell annotations, colorbar, and triple publishing export (PNG, SVG, PDF)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "activity.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_chart(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    days = df["day"].tolist()
    time_blocks = [c for c in df.columns if c != "day"]
    matrix = df[time_blocks].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0)

    # Label ticks
    ax.set_xticks(range(len(time_blocks)), labels=time_blocks, fontweight="semibold")
    ax.set_yticks(range(len(days)), labels=days, fontweight="semibold")

    # Annotate numeric values inside each heatmap cell
    vmax = matrix.max()
    for row in range(len(days)):
        for col in range(len(time_blocks)):
            val = matrix[row, col]
            text_color = "white" if val > (vmax * 0.65) else "#0f172a"
            ax.text(col, row, f"{val}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)

    cbar = fig.colorbar(image, ax=ax, label="Hourly Activity Events")
    ax.set_title("Weekly User Engagement Heatmap (Synthetic)", fontsize=13, fontweight="bold")

    png_path = output_dir / "05-activity-heatmap.png"
    svg_path = output_dir / "05-activity-heatmap.svg"
    pdf_path = output_dir / "05-activity-heatmap.pdf"

    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}, {svg_path}, and {pdf_path}")


if __name__ == "__main__":
    generate_chart()
