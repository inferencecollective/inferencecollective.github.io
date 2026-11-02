"""Lab 5: same business calculation in pandas and SQL (SQLite in memory)."""
import sqlite3
import pandas as pd
from labs.common import load_orders, clean_orders, valid_paid_orders
from labs.lab03_sql_joins import load_customers


def pandas_segment_revenue() -> pd.DataFrame:
    paid = valid_paid_orders(clean_orders(load_orders()))
    customers = load_customers()
    joined = paid.merge(customers, on="customer_id", how="inner", validate="many_to_one")
    return (joined.groupby("segment", as_index=False)
            .agg(revenue=("amount", "sum"))
            .sort_values("segment", ignore_index=True))


def sqlite_segment_revenue() -> pd.DataFrame:
    paid = valid_paid_orders(clean_orders(load_orders()))
    customers = load_customers()
    with sqlite3.connect(":memory:") as conn:
        paid.to_sql("orders", conn, index=False, if_exists="replace")
        customers.to_sql("customers", conn, index=False, if_exists="replace")
        sql = """
            SELECT c.segment, SUM(o.amount) AS revenue
            FROM orders AS o
            INNER JOIN customers AS c ON o.customer_id = c.customer_id
            GROUP BY c.segment
            ORDER BY c.segment
        """
        result = pd.read_sql_query(sql, conn)
    return result


def compare_results() -> bool:
    a, b = pandas_segment_revenue(), sqlite_segment_revenue()
    return (a["segment"].tolist() == b["segment"].tolist()
            and all(abs(x - y) < 1e-9 for x, y in zip(a["revenue"], b["revenue"])))


if __name__ == "__main__":
    print("pandas:\n", pandas_segment_revenue().to_string(index=False))
    print("\nSQL:\n", sqlite_segment_revenue().to_string(index=False))
    print("\nSame results?", compare_results())
