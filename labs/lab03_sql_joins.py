"""Lab 3: explicit SQL-style joins, missing matches and key cardinality."""
from pathlib import Path
import pandas as pd
from labs.common import DATA_DIR, load_orders, clean_orders


def load_customers(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "customers.csv", dtype={"customer_id": "string"})


def enriched_orders(customers: pd.DataFrame | None = None) -> pd.DataFrame:
    orders = clean_orders(load_orders())
    customers = load_customers() if customers is None else customers
    # Many orders may belong to one customer, never duplicate the order rows.
    return orders.merge(customers, on="customer_id", how="left",
                        validate="many_to_one", indicator=True)


def unmatched_orders() -> pd.DataFrame:
    joined = enriched_orders()
    return joined.loc[joined["_merge"].eq("left_only"), ["order_id", "customer_id"]].copy()


if __name__ == "__main__":
    print("Join row count:", len(enriched_orders()))
    print("Orders without a customer record:\n", unmatched_orders().to_string(index=False))
