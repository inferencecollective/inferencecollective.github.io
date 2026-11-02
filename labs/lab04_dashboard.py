"""Lab 4: 4-panel executive dashboard via subplot_mosaic."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_chart(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    growth_df = pd.read_csv(data_dir / "growth.csv")
    growth_df["date"] = pd.to_datetime(growth_df["date"])
    channels_df = pd.read_csv(data_dir / "channels.csv").sort_values("leads", ascending=True)

    rng = np.random.default_rng(42)
    order_sizes = rng.gamma(shape=3.0, scale=25.0, size=500)
    campaign_spend = rng.uniform(200, 1500, size=24)
    campaign_orders = (campaign_spend * 0.45) + rng.normal(0, 40, size=24)

    fig, axd = plt.subplot_mosaic(
        [["trend", "channels"], ["relationship", "distribution"]],
        figsize=(11, 7.5),
        layout="constrained"
    )

    # 1. Trend: Monthly Visitors & Orders
    axd["trend"].plot(growth_df["date"], growth_df["visitors"], marker="o", color="#2563eb", label="Visitors")
    axd["trend"].plot(growth_df["date"], growth_df["orders"], marker="s", color="#059669", label="Orders")
    axd["trend"].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axd["trend"].set_title("A. Monthly Visitors & Orders", fontsize=11, fontweight="bold")
    axd["trend"].set_ylabel("Monthly Count")
    axd["trend"].grid(axis="y", alpha=0.2)
    axd["trend"].legend(frameon=False, fontsize=9)

    # 2. Categories: Acquisition Channels
    bars = axd["channels"].barh(channels_df["channel"], channels_df["leads"], color="#3b82f6")
    axd["channels"].bar_label(bars, fmt="%d", padding=4, fontsize=9)
    axd["channels"].set_title("B. Leads by Acquisition Channel", fontsize=11, fontweight="bold")
    axd["channels"].set_xlabel("Qualified Leads")
    axd["channels"].set_xlim(left=0, right=channels_df["leads"].max() * 1.2)
    axd["channels"].grid(axis="x", alpha=0.2)

    # 3. Relationship: Spend vs Orders
    axd["relationship"].scatter(campaign_spend, campaign_orders, color="#8b5cf6", alpha=0.8, s=45)
    axd["relationship"].set_title("C. Campaign Spend vs New Orders", fontsize=11, fontweight="bold")
    axd["relationship"].set_xlabel("Campaign Spend (USD)")
    axd["relationship"].set_ylabel("Orders Generated")
    axd["relationship"].grid(alpha=0.2)

    # 4. Distribution: Order Basket Sizes
    axd["distribution"].hist(order_sizes, bins=20, color="#f59e0b", edgecolor="white", alpha=0.85)
    axd["distribution"].axvline(np.median(order_sizes), color="#b45309", linestyle="--", linewidth=1.5, label=f"Median: ${np.median(order_sizes):.0f}")
    axd["distribution"].set_title("D. Order Basket Value Distribution", fontsize=11, fontweight="bold")
    axd["distribution"].set_xlabel("Order Value (USD)")
    axd["distribution"].set_ylabel("Frequency")
    axd["distribution"].grid(axis="y", alpha=0.2)
    axd["distribution"].legend(frameon=False, fontsize=9)

    fig.suptitle("Executive Analytics Dashboard (Synthetic Demonstration)", fontsize=14, fontweight="bold")

    png_path = output_dir / "04-four-panel-dashboard.png"
    svg_path = output_dir / "04-four-panel-dashboard.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    generate_chart()
