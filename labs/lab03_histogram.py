"""Lab 3: Skewed wait time distribution with histogram, mean, and median reference lines."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_chart(output_dir: Path = OUTPUT_DIR, seed: int = 42, size: int = 600):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    wait_minutes = rng.lognormal(mean=1.2, sigma=0.55, size=size)

    mean_val = float(np.mean(wait_minutes))
    median_val = float(np.median(wait_minutes))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    n, bins, patches = ax.hist(wait_minutes, bins=25, color="#60a5fa", edgecolor="white", alpha=0.85)

    ax.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} min")
    ax.axvline(median_val, color="#16a34a", linestyle=":", linewidth=2.5, label=f"Median: {median_val:.2f} min")

    ax.set_title("Customer Support Wait Time Distribution (Synthetic)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wait Time (minutes)")
    ax.set_ylabel("Frequency (Count)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    png_path = output_dir / "03-wait-time-distribution.png"
    svg_path = output_dir / "03-wait-time-distribution.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_chart()
