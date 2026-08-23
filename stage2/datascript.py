"""
fetch_data.py
Downloads daily VIX and S&P 500 data from Yahoo Finance and saves raw CSVs.
Unified sample period: Jan 1990 - present (2026), covering the full span
referenced across both Stage 1 papers.
"""

import yfinance as yf
import pandas as pd

START_DATE = "1990-01-01"
END_DATE = None  # None = up to today

def fetch(ticker, name):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    df.to_csv(f"data/raw/{name}_raw.csv")
    print(f"{name}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    return df

if __name__ == "__main__":
    vix = fetch("^VIX", "vix")
    sp500 = fetch("^GSPC", "sp500")