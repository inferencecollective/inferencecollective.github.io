"""Lab 1: CSV cleaning and quality checks. No real customer data is used."""
from labs.common import load_orders, clean_orders


def data_quality_summary():
    orders = clean_orders(load_orders())
    return {
        "rows": len(orders),
        "invalid_dates": int(orders["order_date"].isna().sum()),
        "invalid_amounts": int(orders["amount"].isna().sum()),
        "status_counts": orders["status"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    print("CSV quality report:")
    print(data_quality_summary())
    print("\nCleaned preview:")
    print(clean_orders(load_orders()).head().to_string(index=False))
