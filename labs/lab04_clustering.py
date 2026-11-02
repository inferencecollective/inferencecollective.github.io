"""Lab 4: Unsupervised clustering - KMeans, silhouette scoring, and 2D PCA projection."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_lab(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    X, _ = make_blobs(n_samples=240, n_features=5, centers=3, cluster_std=1.4, random_state=42)

    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, clusters)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Silhouette Score (3 clusters): {sil:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
        s=45
    )

    # Plot Centroids
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=160,
        color="#dc2626",
        edgecolor="black",
        linewidth=1.5,
        label="Cluster Centroids"
    )

    ev_sum = pca.explained_variance_ratio_.sum()
    ax.set_title(
        f"KMeans Clusters Projected via 2D PCA\n(Silhouette: {sil:.3f}, Total Explained Variance: {ev_sum:.1%})",
        fontsize=11,
        fontweight="bold"
    )
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    png_path = output_dir / "04-sklearn-pca-clusters.png"
    svg_path = output_dir / "04-sklearn-pca-clusters.svg"
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    run_lab()
