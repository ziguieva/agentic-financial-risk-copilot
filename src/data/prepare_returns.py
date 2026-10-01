from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "market_prices.csv"

OUTPUT_RETURNS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "market_returns.csv"
)

OUTPUT_STATS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "returns_statistics.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_prices() -> pd.DataFrame:
    """
    Load cleaned market prices.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Price file not found: {INPUT_FILE}"
        )

    prices = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True,
    )

    prices = prices.sort_index()

    print("Price data loaded successfully.")
    print(f"Observations: {len(prices)}")
    print(f"Assets: {len(prices.columns)}")

    return prices


# ============================================================
# RETURNS
# ============================================================

def compute_log_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily logarithmic returns.

    Formula:
        r_t = ln(P_t / P_{t-1})
    """

    returns = np.log(
        prices / prices.shift(1)
    )

    returns = returns.dropna()

    return returns


# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each asset.
    """

    results = []

    for asset in returns.columns:

        series = returns[asset].dropna()

        jb_result = jarque_bera(series)

        mean_daily = series.mean()

        volatility_daily = series.std()

        volatility_annualized = (
            volatility_daily * np.sqrt(252)
        )

        results.append(
            {
                "asset": asset,
                "observations": len(series),
                "mean_daily": mean_daily,
                "std_daily": volatility_daily,
                "volatility_annualized": volatility_annualized,
                "min_return": series.min(),
                "max_return": series.max(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "jarque_bera_stat": jb_result.statistic,
                "jarque_bera_pvalue": jb_result.pvalue,
            }
        )

    statistics = pd.DataFrame(results)

    statistics = statistics.set_index("asset")

    return statistics


# ============================================================
# SAVE
# ============================================================

def save_results(
    returns: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """
    Save returns and statistical summary.
    """

    returns.to_csv(
        OUTPUT_RETURNS_FILE
    )

    statistics.to_csv(
        OUTPUT_STATS_FILE
    )

    print("\nFiles saved:")

    print(
        f"- {OUTPUT_RETURNS_FILE}"
    )

    print(
        f"- {OUTPUT_STATS_FILE}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    returns: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """
    Display key information in the terminal.
    """

    print("\n========================================")
    print("RETURNS DATA SUMMARY")
    print("========================================")

    print(
        f"Start date : "
        f"{returns.index.min().date()}"
    )

    print(
        f"End date   : "
        f"{returns.index.max().date()}"
    )

    print(
        f"Rows       : "
        f"{len(returns)}"
    )

    print("\nFirst returns:")
    print(returns.head())

    print("\n========================================")
    print("STATISTICAL SUMMARY")
    print("========================================")

    columns_to_display = [
        "mean_daily",
        "volatility_annualized",
        "skewness",
        "kurtosis",
        "jarque_bera_pvalue",
    ]

    print(
        statistics[
            columns_to_display
        ].round(6)
    )

    print("\n========================================")
    print("NORMALITY TEST")
    print("========================================")

    for asset in statistics.index:

        pvalue = statistics.loc[
            asset,
            "jarque_bera_pvalue",
        ]

        if pvalue < 0.05:
            conclusion = (
                "returns are NOT normally distributed"
            )
        else:
            conclusion = (
                "normality cannot be rejected"
            )

        print(
            f"{asset}: "
            f"p-value = {pvalue:.6g} "
            f"-> {conclusion}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    prices = load_prices()

    returns = compute_log_returns(
        prices
    )

    statistics = compute_statistics(
        returns
    )

    save_results(
        returns,
        statistics,
    )

    print_summary(
        returns,
        statistics,
    )

    print(
        "\nReturns preparation completed successfully."
    )


if __name__ == "__main__":
    main()