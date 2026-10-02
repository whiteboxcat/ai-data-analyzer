"""Generate a deliberately messy multi-sheet workbook for testing.

    python tests/make_sample.py   ->  tests/sample_messy.xlsx + tests/sample_sales.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent

# --- Sheet 1: Sales (messy) -------------------------------------------------
n = 400
dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")
cats = rng.choice(["Dress", "T-Shirt", "Hijab", "Shoes", "Bag"], n, p=[.25, .3, .2, .15, .1])
channel = rng.choice(["Shopee", "Tokopedia", "TikTok Shop", "Website"], n, p=[.4, .25, .25, .1])
qty = rng.integers(1, 6, n)
price = np.select([cats == "Dress", cats == "Shoes", cats == "Bag"], [250_000, 400_000, 350_000], 120_000)
gmv = qty * price
discount = (gmv * rng.choice([0, 0.05, 0.1, 0.2], n)).round(-3)

sales = pd.DataFrame({
    "Order No": [f"INV-{100000 + i}" for i in range(n)],
    # mixed formats, dd/mm/yyyy (Indonesian style)
    "Order Date": [d.strftime("%d/%m/%Y") if i % 3 else d.strftime("%Y-%m-%d") for i, d in enumerate(dates)],
    "Category": cats,
    "Channel": channel,
    "City": rng.choice(["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar"], n),
    "Qty": qty,
    # GMV stored as Indonesian currency text
    "GMV": [f"Rp {v:,.0f}".replace(",", ".") for v in gmv],
    "Discount": discount,
    "Net": gmv - discount,
    "Customer Note": rng.choice(["", "kirim cepat ya", "gift wrap", None, "warna sesuai foto"], n),
})
# inject issues
sales.loc[rng.choice(n, 15, replace=False), "Category"] = (sales["Category"].sample(15, random_state=1).str.lower() + "  ").values
sales.loc[rng.choice(n, 12, replace=False), "City"] = None
sales.loc[rng.choice(n, 8, replace=False), "Discount"] = np.nan
sales.loc[5, "Net"] = 98_000_000  # outlier / typo
sales.loc[7, "Qty"] = -2          # impossible negative
sales = pd.concat([sales, sales.iloc[[10, 11, 12]]], ignore_index=True)  # duplicate rows
sales["Empty Col"] = None

# --- Sheet 2: Daily summary with YYYYMMDD ints -----------------------------
days = pd.date_range("2025-03-01", periods=60, freq="D")
daily = pd.DataFrame({
    "Date Key": days.strftime("%Y%m%d").astype(int),
    "Visitors": rng.integers(800, 2500, 60),
    "Orders": rng.integers(20, 120, 60),
    "Ad Spend": rng.integers(500_000, 3_000_000, 60),
    "Store": "Main Store",
})

# --- Sheet 3: Products ------------------------------------------------------
products = pd.DataFrame({
    "SKU": [f"SKU{1000 + i}" for i in range(40)],
    "Product Name": [f"Product {i}" for i in range(40)],
    "Category": rng.choice(["Dress", "T-Shirt", "Hijab", "Shoes", "Bag"], 40),
    "Price": rng.integers(80, 500, 40) * 1000,
    "Stock": rng.integers(0, 300, 40),
    "Rating": rng.choice([3, 4, 5], 40),
})

with pd.ExcelWriter(OUT / "sample_messy.xlsx") as xw:
    # Title rows above the header, like many real exports
    title = pd.DataFrame([["Sales Report Jan-Sep 2025"], ["Exported from marketplace dashboard"], [None]])
    title.to_excel(xw, sheet_name="Sales", index=False, header=False)
    sales.to_excel(xw, sheet_name="Sales", index=False, startrow=3)
    daily.to_excel(xw, sheet_name="Daily Traffic", index=False)
    products.to_excel(xw, sheet_name="Products", index=False)
    pd.DataFrame().to_excel(xw, sheet_name="Blank", index=False)

sales.drop(columns=["Empty Col"]).to_csv(OUT / "sample_sales.csv", index=False, sep=";")
print("written:", OUT / "sample_messy.xlsx", OUT / "sample_sales.csv")
