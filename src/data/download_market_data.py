from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = {
    "BNP": "BNP.PA",
    "SOCIETE_GENERALE": "GLE.PA",
    "AXA": "CS.PA",
    "CAC40": "^FCHI",
    "SP500": "^GSPC",
}

START_DATE = "2015-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DATA_DIR / "market_data_raw.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "market_prices.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def create_directories() -> None:
    """
    Create data directories if they do not already exist.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_market_data() -> pd.DataFrame:
    """
    Download historical market data from Yahoo Finance.
    """

    print("Downloading market data...")

    ticker_list = list(TICKERS.values())

    data = yf.download(
        ticker_list,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data downloaded from Yahoo Finance.")

    print(f"Downloaded rows: {len(data)}")

    return data


def extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract adjusted closing prices and rename columns
    using readable asset names.
    """

    if "Close" not in data.columns.get_level_values(0):
        raise ValueError("Close prices not found in downloaded data.")

    prices = data["Close"].copy()

    reverse_mapping = {
        yahoo_ticker: asset_name
        for asset_name, yahoo_ticker in TICKERS.items()
    }

    prices = prices.rename(columns=reverse_mapping)

    prices.index = pd.to_datetime(prices.index)

    prices = prices.sort_index()

    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data.

    Main operations:
    - remove completely empty rows
    - forward-fill occasional missing observations
    - remove remaining missing values
    """

    print("\nMissing values before cleaning:")
    print(prices.isna().sum())

    prices = prices.dropna(how="all")

    prices = prices.ffill()

    prices = prices.dropna()

    print("\nMissing values after cleaning:")
    print(prices.isna().sum())

    return prices


def save_data(
    raw_data: pd.DataFrame,
    clean_prices_data: pd.DataFrame,
) -> None:
    """
    Save raw and cleaned datasets.
    """

    raw_data.to_csv(RAW_FILE)

    clean_prices_data.to_csv(PROCESSED_FILE)

    print(f"\nRaw data saved to:")
    print(RAW_FILE)

    print(f"\nProcessed prices saved to:")
    print(PROCESSED_FILE)


def print_summary(prices: pd.DataFrame) -> None:
    """
    Display a basic summary of the dataset.
    """

    print("\n========================================")
    print("MARKET DATA SUMMARY")
    print("========================================")

    print(f"Number of observations : {len(prices)}")
    print(f"Number of assets       : {len(prices.columns)}")

    print(f"Start date             : {prices.index.min().date()}")
    print(f"End date               : {prices.index.max().date()}")

    print("\nAssets:")
    for column in prices.columns:
        print(f"- {column}")

    print("\nFirst rows:")
    print(prices.head())

    print("\nLast rows:")
    print(prices.tail())


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    create_directories()

    raw_data = download_market_data()

    prices = extract_close_prices(raw_data)

    prices = clean_prices(prices)

    save_data(
        raw_data=raw_data,
        clean_prices_data=prices,
    )

    print_summary(prices)

    print("\nData pipeline completed successfully.")


if __name__ == "__main__":
    main()