"""Dataset generation and categorical preparation for CatBoost tutorials and labs."""
import numpy as np
import pandas as pd

CATEGORICAL = ["region", "plan", "product_type"]

def make_customers(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    regions = rng.choice(["North", "South", "East", "West", np.nan], size=n, p=[0.28, 0.28, 0.22, 0.18, 0.04])
    plans = rng.choice(["basic", "plus", "premium"], size=n, p=[0.5, 0.3, 0.2])
    product_types = rng.choice(["fiber", "dsl", "wireless"], size=n, p=[0.45, 0.35, 0.20])
    
    tenure = rng.integers(1, 72, size=n)
    monthly_charges = np.where(plans == "basic", 35.0, np.where(plans == "plus", 70.0, 110.0)) + rng.normal(0, 5, size=n)
    nan_mask = rng.random(size=n) < 0.03
    monthly_charges[nan_mask] = np.nan
    
    support_calls = rng.integers(0, 6, size=n)
    
    log_odds = (
        -1.8
        + 0.02 * np.nan_to_num(monthly_charges, nan=50.0)
        - 0.04 * tenure
        + 0.35 * support_calls
        + (plans == "basic") * 0.4
    )
    prob = 1 / (1 + np.exp(-log_odds))
    churned = (rng.random(size=n) < prob).astype(int)
    
    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": np.round(monthly_charges, 2),
        "support_calls": support_calls,
        "region": regions,
        "plan": plans,
        "product_type": product_types,
        "churned": churned,
    })

def prepare_categories(df: pd.DataFrame, cat_cols=CATEGORICAL, missing_token: str = "__MISSING__") -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].fillna(missing_token).astype(str)
    return out
