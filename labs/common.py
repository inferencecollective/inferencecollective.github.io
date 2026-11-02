"""Shared loading and validation. Run from any working directory."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
PREVIEWS_DIR = Path(__file__).resolve().parent.parent / "previews"

DATA = DATA_DIR
OUTPUTS = OUTPUTS_DIR
PREVIEWS = PREVIEWS_DIR


def load_orders(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw CSV with identifiers preserved as strings."""
    csv_path = Path(path) if path is not None else DATA_DIR / "orders.csv"
    return pd.read_csv(csv_path, dtype={"order_id": "string", "customer_id": "string"})


def clean_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize data without mutating the caller's DataFrame."""
    required = {"order_id", "customer_id", "order_date", "amount", "quantity", "region", "channel", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame.copy()
    result["order_date"] = pd.to_datetime(result["order_date"], format="%Y-%m-%d", errors="coerce")
    cleaned_amount = (result["amount"].astype("string")
                      .str.replace("$", "", regex=False)
                      .str.replace(",", "", regex=False)
                      .str.strip())
    result["amount"] = pd.to_numeric(cleaned_amount, errors="coerce")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    result["status"] = result["status"].astype("string").str.strip().str.lower()
    result["region"] = result["region"].astype("string").str.strip().str.title()
    result["channel"] = result["channel"].astype("string").str.strip().str.lower()
    return result


def valid_paid_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Business rule: revenue counts valid paid orders only, not pending/refunded."""
    keep = (frame["status"].eq("paid").fillna(False)
            & frame["order_date"].notna()
            & frame["amount"].gt(0).fillna(False)
            & frame["quantity"].gt(0).fillna(False))
    return frame.loc[keep].copy()


def classification_data(n_samples: int = 400, random_state: int = 42):
    """Generate synthetic binary classification data."""
    from sklearn.datasets import make_classification
    return make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        weights=[0.75, 0.25],
        random_state=random_state
    )


def regression_data(n_samples: int = 400, random_state: int = 42):
    """Generate synthetic regression data."""
    from sklearn.datasets import make_regression
    return make_regression(
        n_samples=n_samples,
        n_features=6,
        noise=12.0,
        random_state=random_state
    )


def churn_data(path: str | Path | None = None):
    """Load synthetic churn data with native categorical columns."""
    csv_path = Path(path) if path is not None else DATA_DIR / "customers_churn.csv"
    df = pd.read_csv(csv_path)
    for col in ["plan", "contract"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    X = df.drop(columns=["customer_id", "churn"], errors="ignore")
    y = df["churn"]
    return X, y


def load_split(split: str = "train", augment: bool = False, root: str | Path | None = None):
    """Load a split from synthetic_shapes as a torchvision ImageFolder."""
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import v2
    import torch

    data_root = Path(root) if root is not None else DATA_DIR / "synthetic_shapes"
    split_dir = data_root / split
    if not split_dir.exists():
        from data.generate_samples import generate_all
        generate_all()

    if augment and split == "train":
        transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.RandomRotation(degrees=10),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    return ImageFolder(str(split_dir), transform=transform)


def load_tf_regression(path: str | Path | None = None):
    """Load synthetic 1D regression data for GradientTape."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "tf-regression.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return data["x"], data["y"], float(data["true_w"]), float(data["true_b"])


def load_tf_moons(path: str | Path | None = None):
    """Load synthetic 2D non-linear classification data."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "tf-moons.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def load_tf_shapes(path: str | Path | None = None):
    """Load synthetic 28x28 grayscale shapes (circles vs squares) in NHWC format."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "tf-shapes-28x28.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def load_keras_regression(path: str | Path | None = None):
    """Load synthetic 1D regression data: y ≈ 2.5x - 0.8 + noise."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "keras-regression.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        float(data["true_slope"]),
        float(data["true_intercept"])
    )


def load_keras_circles(path: str | Path | None = None):
    """Load synthetic 2D concentric circles data with Train/Val/Test splits."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "keras-circles.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"]
    )


def load_keras_bars(path: str | Path | None = None):
    """Load synthetic 24x24 grayscale bar patterns in channels-last (24, 24, 1)."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "keras-bars-24x24.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def load_keras_sequences(path: str | Path | None = None):
    """Load synthetic token integer sequences of shape (batch, 8)."""
    import numpy as np
    npz_path = Path(path) if path is not None else DATA_DIR / "keras-sequences.npz"
    if not npz_path.exists():
        from data.generate_samples import generate_all
        generate_all()
    data = np.load(npz_path)
    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        int(data["target_token"]),
        int(data["vocab_size"])
    )



