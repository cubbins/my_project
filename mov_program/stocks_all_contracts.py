import yfinance as yf
import pandas as pd

# ---------------------------------------------------------
# 1. Define ALL continuous futures tickers
# ---------------------------------------------------------
contracts = {
    # Energy
    "CrudeOil": "CL=F",
    "BrentCrude": "BZ=F",
    "NaturalGas": "NG=F",
    "RBOBGasoline": "RB=F",
    "HeatingOil": "HO=F",

    # Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",

    # Agriculture
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "SoybeanOil": "ZL=F",
    "SoybeanMeal": "ZM=F",
    "LiveCattle": "LE=F",

    # Currencies
    "EuroFX": "6E=F",
    "JapaneseYen": "6J=F",
    "BritishPound": "6B=F",
    "AustralianDollar": "6A=F",
    "CanadianDollar": "6C=F",

    # Interest Rates / Treasuries
    "Eurodollar": "GE=F",
    "UST2Y": "ZT=F",
    "UST5Y": "ZF=F",
    "UST10Y": "ZN=F",
    "UST30Y": "ZB=F",

    # Equity Index Futures
    "EMiniDow": "YM=F",
    "EMiniSP500": "ES=F",
    "EMiniNasdaq100": "NQ=F"
}


import os

# Add this after defining start_date/end_date
output_dir = "futures_data"
os.makedirs(output_dir, exist_ok=True)

start_date = "2025-01-01"
end_date = "2027-01-01"

# ---------------------------------------------------------
# 2. Loop through each contract and process
# ---------------------------------------------------------

count = 0  # Add this before the loop


for name, ticker in contracts.items():





    print(f"\n==============================")
    print(f"Downloading {name} ({ticker})")
    print(f"==============================")

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # Print first rows
    print("\nFIRST ROWS:")
    print(data.head())

    # Print last rows
    print("\nLAST ROWS:")
    print(data.tail())

    # Save to tab-delimited file
    filename = f"{name}.txt"
    
    filename = os.path.join(output_dir, f"{name}.txt")
    data.to_csv(filename, sep="\t")

    print(f"\nSaved {len(data)} rows to {filename}")

    # ... all your existing code ...

    #count += 1
    #if count == 5:
    #    break

