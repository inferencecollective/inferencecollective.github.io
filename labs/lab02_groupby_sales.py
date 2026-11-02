"""Lab 2: GroupBy KPIs and monthly revenue."""
import pandas as pd
from labs.common import load_orders, clean_orders, valid_paid_orders


def revenue_by_region() -> pd.DataFrame:
    paid = valid_paid_orders(clean_orders(load_orders()))
    return (paid.groupby("region", as_index=False)
            .agg(orders=("order_id", "nunique"), revenue=("amount", "sum"))
            .sort_values("revenue", ascending=False, ignore_index=True))


def monthly_revenue() -> pd.DataFrame:
    paid = valid_paid_orders(clean_orders(load_orders()))
    result = (paid.set_index("order_date").sort_index()
              .resample("MS")["amount"].sum().rename("revenue").reset_index())
    return result


if __name__ == "__main__":
    print("Revenue by region:\n", revenue_by_region().to_string(index=False))
    print("\nMonthly revenue:\n", monthly_revenue().to_string(index=False))
