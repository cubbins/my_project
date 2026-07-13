#!/usr/bin/env python3
"""
Export AAPL option chain data from yfinance to a tab-delimited text file.

This program:
1. Downloads the available option expiration dates for a ticker.
2. Selects an expiration date (first available by default, or user-specified).
3. Downloads both calls and puts for that expiration.
4. Writes them to a tab-delimited text file with no header and no index.

Usage examples:
    python export_options.py
    python export_options.py --ticker AAPL
    python export_options.py --ticker AAPL --expiration 2026-04-17
    python export_options.py --ticker AAPL --output aapl_options.txt
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download option chain data with yfinance and export to a tab-delimited text file."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol to download options for (default: AAPL).",
    )
    parser.add_argument(
        "--expiration",
        default=None,
        help=(
            "Expiration date in YYYY-MM-DD format. "
            "If omitted, the first available expiration is used."
        ),
    )
    parser.add_argument(
        "--output",
        default="options.txt",
        help="Output filename (default: options.txt).",
    )
    return parser.parse_args()


def choose_expiration(ticker_obj: yf.Ticker, requested_expiration: Optional[str]) -> str:
    expirations = list(ticker_obj.options)

    if not expirations:
        raise RuntimeError("No option expiration dates were returned for this ticker.")

    if requested_expiration is None:
        return expirations[0]

    if requested_expiration not in expirations:
        available = ", ".join(expirations)
        raise ValueError(
            f"Requested expiration '{requested_expiration}' is not available.\n"
            f"Available expirations: {available}"
        )

    return requested_expiration


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

    # Normalize datetime columns if present
    if "lastTradeDate" in combined.columns:
        combined["lastTradeDate"] = pd.to_datetime(
            combined["lastTradeDate"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        combined["lastTradeDate"] = combined["lastTradeDate"].fillna("")

    # Create a fixed-column output similar to your stock export style
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


def main() -> int:
    args = parse_args()
    ticker_symbol = args.ticker.strip().upper()

    if not ticker_symbol:
        print("Error: ticker symbol must not be empty.", file=sys.stderr)
        return 1

    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        expiration = choose_expiration(ticker_obj, args.expiration)

        option_chain = ticker_obj.option_chain(expiration)
        calls = option_chain.calls
        puts = option_chain.puts

        if calls.empty and puts.empty:
            raise RuntimeError(
                f"No option chain rows were returned for {ticker_symbol} {expiration}."
            )

        final_data = build_output_dataframe(calls, puts, ticker_symbol, expiration)

        final_data.to_csv(args.output, sep="\t", header=False, index=False)

        print(f"Ticker: {ticker_symbol}")
        print(f"Expiration used: {expiration}")
        print(f"Calls rows: {len(calls)}")
        print(f"Puts rows: {len(puts)}")
        print(f"Total rows written: {len(final_data)}")
        print(f"Output file: {args.output}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
