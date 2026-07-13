#!/usr/bin/env python3
"""
Export option chain data from yfinance to a tab-delimited text file.

This version:
- Selects NEXT MONTH expiration by default (not current expiry)
- Still allows manual override via --expiration
- Outputs fixed-column format compatible with your C#/processing pipeline
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download option chain data with yfinance and export to a tab-delimited text file."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--expiration",
        default=None,
        help="Optional expiration date YYYY-MM-DD (overrides auto selection)",
    )
    parser.add_argument(
        "--output",
        default="options.txt",
        help="Output filename (default: options.txt)",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Expiration selection (NEXT MONTH logic)
# ------------------------------------------------------------
def choose_expiration(ticker_obj: yf.Ticker, requested_expiration: Optional[str]) -> str:
    expirations = list(ticker_obj.options)

    if not expirations:
        raise RuntimeError("No option expiration dates were returned.")

    # If user explicitly specifies expiration, use it
    if requested_expiration is not None:
        if requested_expiration not in expirations:
            raise ValueError(
                f"Expiration '{requested_expiration}' not available.\n"
                f"Available: {', '.join(expirations)}"
            )
        return requested_expiration

    # ---- AUTO: select NEXT MONTH ----
    today = date.today()

    if today.month == 12:
        target_year = today.year + 1
        target_month = 1
    else:
        target_year = today.year
        target_month = today.month + 1

    parsed = [
        (datetime.strptime(exp, "%Y-%m-%d").date(), exp)
        for exp in expirations
    ]

    # Find first expiration in next month
    for exp_date, exp_str in parsed:
        if exp_date.year == target_year and exp_date.month == target_month:
            return exp_str

    raise RuntimeError(
        f"No expiration found for next month ({target_year}-{target_month:02d}).\n"
        f"Available expirations: {', '.join(expirations)}"
    )


# ------------------------------------------------------------
# Data formatting
# ------------------------------------------------------------
def build_output_dataframe(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    ticker: str,
    expiration: str,
) -> pd.DataFrame:

    calls = calls.copy()
    puts = puts.copy()

    calls["optionType"] = "CALL"
    puts["optionType"] = "PUT"

    combined = pd.concat([calls, puts], ignore_index=True)

    # Normalize datetime
    if "lastTradeDate" in combined.columns:
        combined["lastTradeDate"] = pd.to_datetime(
            combined["lastTradeDate"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        combined["lastTradeDate"] = combined["lastTradeDate"].fillna("")

    # Fixed column mapping
    final_data = pd.DataFrame()
    final_data[0] = ticker.upper()
    final_data[1] = expiration
    final_data[2] = combined.get("optionType", "")
    final_data[3] = combined.get("contractSymbol", "")
    final_data[4] = combined.get("lastTradeDate", "")
    final_data[5] = combined.get("strike", "")
    final_data[6] = combined.get("lastPrice", "")
    final_data[7] = combined.get("bid", "")
    final_data[8] = combined.get("ask", "")
    final_data[9] = combined.get("change", "")
    final_data[10] = combined.get("percentChange", "")
    final_data[11] = combined.get("volume", "")
    final_data[12] = combined.get("openInterest", "")
    final_data[13] = combined.get("impliedVolatility", "")
    final_data[14] = combined.get("inTheMoney", "")
    final_data[15] = combined.get("currency", "")

    return final_data


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main() -> int:
    args = parse_args()
    ticker_symbol = args.ticker.strip().upper()

    if not ticker_symbol:
        print("Error: ticker must not be empty.", file=sys.stderr)
        return 1

    try:
        ticker_obj = yf.Ticker(ticker_symbol)

        expiration = choose_expiration(ticker_obj, args.expiration)

        chain = ticker_obj.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            raise RuntimeError("No option data returned.")

        final_data = build_output_dataframe(
            calls, puts, ticker_symbol, expiration
        )

        final_data.to_csv(args.output, sep="\t", header=False, index=False)

        print(f"Ticker: {ticker_symbol}")
        print(f"Expiration selected: {expiration}")
        print(f"Calls: {len(calls)}, Puts: {len(puts)}")
        print(f"Rows written: {len(final_data)}")
        print(f"Output: {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
