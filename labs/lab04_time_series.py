"""Lab 4: daily sales, seven-day trailing totals and monthly resampling."""
import pandas as pd
from labs.common import load_orders, clean_orders, valid_paid_orders


def daily_revenue() -> pd.Series:
    paid = valid_paid_orders(clean_orders(load_orders()))
    daily = paid.groupby("order_date")["amount"].sum().sort_index()
    # Include days with zero paid orders; do not fill unknown transactions as revenue.
    return daily.asfreq("D", fill_value=0).rename("daily_revenue")


def trailing_seven_day_revenue() -> pd.Series:
    return daily_revenue().rolling(window=7, min_periods=1).sum().rename("trailing_7d_revenue")


if __name__ == "__main__":
    daily = daily_revenue()
    trailing = trailing_seven_day_revenue()
    print(pd.concat([daily, trailing], axis=1).head(12).to_string())
