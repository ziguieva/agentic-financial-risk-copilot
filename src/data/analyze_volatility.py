from pathlib import Path

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "volatility_diagnostics.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load daily logarithmic returns.
    """

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {RETURNS_FILE}"
        )

    returns = pd.read_csv(
        RETURNS_FILE,
        index_col=0,
        parse_dates=True,
    )

    returns = returns.sort_index()

    print("Returns loaded successfully.")
    print(f"Observations: {len(returns)}")
    print(f"Assets: {len(returns.columns)}")

    return returns


# ============================================================
# ADF TEST
# ============================================================

def run_adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.

    H0:
        The series contains a unit root
        and is therefore non-stationary.

    If p-value < 0.05:
        We reject H0 and consider
        the series stationary.
    """

    result = adfuller(
        series.dropna(),
        autolag="AIC",
    )

    return {
        "adf_statistic": result[0],
        "adf_pvalue": result[1],
    }


# ============================================================
# LJUNG-BOX TEST
# ============================================================

def run_ljung_box_test(
    series: pd.Series,
    lag: int = 20,
) -> dict:
    """
    Ljung-Box test on raw returns.

    H0:
        No autocorrelation up to the selected lag.
    """

    result = acorr_ljungbox(
        series.dropna(),
        lags=[lag],
        return_df=True,
    )

    return {
        "lb_returns_stat": result["lb_stat"].iloc[0],
        "lb_returns_pvalue": result["lb_pvalue"].iloc[0],
    }


# ============================================================
# LJUNG-BOX ON SQUARED RETURNS
# ============================================================

def run_squared_returns_test(
    series: pd.Series,
    lag: int = 20,
) -> dict:
    """
    Ljung-Box test on squared returns.

    Significant autocorrelation in squared returns
    is evidence of volatility clustering and
    ARCH/GARCH-type effects.
    """

    squared_returns = series.dropna() ** 2

    result = acorr_ljungbox(
        squared_returns,
        lags=[lag],
        return_df=True,
    )

    return {
        "lb_squared_stat": result["lb_stat"].iloc[0],
        "lb_squared_pvalue": result["lb_pvalue"].iloc[0],
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_diagnostics(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all diagnostics for each asset.
    """

    results = []

    for asset in returns.columns:

        series = returns[asset].dropna()

        adf_results = run_adf_test(series)

        lb_returns_results = run_ljung_box_test(
            series,
            lag=20,
        )

        lb_squared_results = run_squared_returns_test(
            series,
            lag=20,
        )

        results.append(
            {
                "asset": asset,
                **adf_results,
                **lb_returns_results,
                **lb_squared_results,
            }
        )

    diagnostics = pd.DataFrame(results)

    diagnostics = diagnostics.set_index("asset")

    return diagnostics


# ============================================================
# INTERPRETATION
# ============================================================

def print_interpretation(
    diagnostics: pd.DataFrame,
) -> None:
    """
    Print statistical interpretation.
    """

    print("\n========================================")
    print("VOLATILITY DIAGNOSTICS")
    print("========================================")

    print(
        diagnostics.round(6)
    )

    print("\n========================================")
    print("INTERPRETATION")
    print("========================================")

    for asset in diagnostics.index:

        adf_pvalue = diagnostics.loc[
            asset,
            "adf_pvalue",
        ]

        returns_pvalue = diagnostics.loc[
            asset,
            "lb_returns_pvalue",
        ]

        squared_pvalue = diagnostics.loc[
            asset,
            "lb_squared_pvalue",
        ]

        print(f"\n{asset}")
        print("-" * len(asset))

        if adf_pvalue < 0.05:
            print(
                "ADF: returns are stationary."
            )
        else:
            print(
                "ADF: stationarity is not confirmed."
            )

        if returns_pvalue < 0.05:
            print(
                "Raw returns: significant autocorrelation detected."
            )
        else:
            print(
                "Raw returns: no significant autocorrelation detected."
            )

        if squared_pvalue < 0.05:
            print(
                "Squared returns: significant autocorrelation detected."
            )
            print(
                "=> Evidence of volatility clustering."
            )
            print(
                "=> ARCH/GARCH modelling is justified."
            )
        else:
            print(
                "Squared returns: no significant autocorrelation detected."
            )


# ============================================================
# SAVE
# ============================================================

def save_results(
    diagnostics: pd.DataFrame,
) -> None:
    """
    Save diagnostics to CSV.
    """

    diagnostics.to_csv(
        OUTPUT_FILE
    )

    print(
        f"\nDiagnostics saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    returns = load_returns()

    diagnostics = compute_diagnostics(
        returns
    )

    save_results(
        diagnostics
    )

    print_interpretation(
        diagnostics
    )

    print(
        "\nVolatility analysis completed successfully."
    )


if __name__ == "__main__":
    main()