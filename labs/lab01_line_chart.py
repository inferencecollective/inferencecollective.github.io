"""Lab 1: Growth line chart with indexed metrics and peak annotation."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "growth.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_chart(data_path: Path = DATA_PATH, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Index both metrics to 100 in January
    df["visitors_indexed"] = (df["visitors"] / df["visitors"].iloc[0]) * 100
    df["orders_indexed"] = (df["orders"] / df["orders"].iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(df["date"], df["visitors_indexed"], marker="o", linewidth=2, label="Visitors (Index Jan=100)", color="#2563eb")
    ax.plot(df["date"], df["orders_indexed"], marker="s", linewidth=2, label="Orders (Index Jan=100)", color="#059669")

    # Annotate peak orders
    peak_idx = df["orders_indexed"].idxmax()
    peak_date = df["date"].iloc[peak_idx]
    peak_val = df["orders_indexed"].iloc[peak_idx]
    peak_raw = df["orders"].iloc[peak_idx]

    ax.annotate(
        f"Peak Orders ({peak_raw:,.0f})",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.Timedelta(days=60), peak_val + 15),
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.5},
        fontweight="bold",
        color="#059669"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Relative Growth Trends: Visitors vs Orders (Synthetic)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Indexed Value (Jan = 100)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    png_path = output_dir / "01-growth-trends.png"
    svg_path = output_dir / "01-growth-trends.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_chart()
