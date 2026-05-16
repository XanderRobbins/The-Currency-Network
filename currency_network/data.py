"""Data download and preprocessing for Currency Network."""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "USDJPY=X",   # invert
    "CHF": "USDCHF=X",   # invert
    "AUD": "AUDUSD=X",
    "CAD": "USDCAD=X",   # invert
    "NZD": "NZDUSD=X",
    "SEK": "USDSEK=X",   # invert
    "NOK": "USDNOK=X",   # invert
}

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"]

INVERT_TICKERS = {"JPY", "CHF", "CAD", "SEK", "NOK"}

START = "2015-01-01"
END = "2024-12-31"


def download_data(data_dir="currency_network/data"):
    """Download FX price data from Yahoo Finance."""
    data_path = Path(data_dir) / "fx_prices.csv"

    if data_path.exists():
        print(f"Loading cached data from {data_path}")
        return pd.read_csv(data_path, index_col=0, parse_dates=True)

    print("Downloading FX data from Yahoo Finance...")
    prices_list = []

    for curr, ticker in TICKERS.items():
        try:
            data = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

            # Extract close prices
            if isinstance(data, pd.DataFrame):
                px = data["Close"].copy()
            else:
                px = data.copy()

            # Convert to Series if needed
            if hasattr(px, 'values'):
                values = px.values
                if values.ndim > 1:
                    values = values.flatten()
                px = pd.Series(values, index=px.index)

            if curr in INVERT_TICKERS:
                px = 1.0 / px

            # Convert to DataFrame for concat
            df_curr = px.to_frame(name=curr)
            prices_list.append(df_curr)
            print(f"  Downloaded {curr} ({ticker})")
        except Exception as e:
            print(f"  Error downloading {curr}: {e}")
            raise

    # Concatenate all currencies
    df = pd.concat(prices_list, axis=1)

    # Add USD = constant 1.0
    df["USD"] = 1.0

    # Reorder columns
    df = df[CURRENCIES]

    # Forward fill, then back fill NaNs
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Drop any remaining NaNs
    df = df.dropna()

    # Save to CSV
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path)
    print(f"Saved {len(df)} days of data to {data_path}")

    return df


def compute_log_returns(prices, data_dir="currency_network/data"):
    """Compute log returns from prices."""
    returns_path = Path(data_dir) / "fx_log_returns.csv"

    if returns_path.exists():
        return pd.read_csv(returns_path, index_col=0, parse_dates=True)

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.to_csv(returns_path)
    print(f"Saved log returns to {returns_path}")

    return log_returns


def compute_volume_proxy(log_returns):
    """
    Use BIS Triennial Survey 2022 FX turnover shares as static volume weights.
    These are the official market share estimates for each currency.
    Source: BIS Triennial Central Bank Survey 2022 (Table 1).
    """
    BIS_WEIGHTS = {
        "USD": 0.884,
        "EUR": 0.315,
        "JPY": 0.167,
        "GBP": 0.129,
        "AUD": 0.067,
        "CAD": 0.062,
        "CHF": 0.052,
        "NZD": 0.021,
        "SEK": 0.020,
        "NOK": 0.017,
    }
    # Build a constant DataFrame with same index as log_returns
    # Values normalized to [0,1]
    weights = np.array([BIS_WEIGHTS[c] for c in log_returns.columns])
    weights = (weights - weights.min()) / (weights.max() - weights.min()) + 1e-6
    volume_df = pd.DataFrame(
        np.tile(weights, (len(log_returns), 1)),
        index=log_returns.index,
        columns=log_returns.columns
    )
    return volume_df
