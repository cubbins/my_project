import yfinance as yf
import pandas as pd
import os

# ---------------------------------------------------------
# 1. Define stock asset
# ---------------------------------------------------------
name = "intc"
ticker = "intc"

output_dir = "stock_data"
os.makedirs(output_dir, exist_ok=True)

start_date = "2025-01-01"
end_date = "2027-01-01"

# ---------------------------------------------------------
# 2. Download MSFT stock data
# ---------------------------------------------------------
print(f"\n==============================")
print(f"Downloading {name} ({ticker})")
print(f"==============================")

data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    auto_adjust=True
)

print("\nFIRST ROWS:")
print(data.head())

print("\nLAST ROWS:")
print(data.tail())

# ---------------------------------------------------------
# 3. Save to tab-delimited file
# ---------------------------------------------------------
filename = os.path.join(output_dir, f"{name}.txt")
data.to_csv(filename, sep="\t")

print(f"\nSaved {len(data)} rows to {filename}")