from pathlib import Path

import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80
LAG = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "garch"
    / "garch_diagnostics.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load logarithmic market returns.
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
    returns = returns.dropna()

    print("Returns loaded successfully.")
    print(f"Observations : {len(returns)}")
    print(f"Assets       : {len(returns.columns)}")

    return returns


# ============================================================
# FIT MODEL
# ============================================================

def fit_model(
    series: pd.Series,
):
    """
    Fit the same AR(1)-GARCH(1,1)-Student-t
    model used in the baseline.

    The model is estimated only on the
    training period.
    """

    series_pct = series * 100.0

    split_index = int(
        len(series_pct) * TRAIN_RATIO
    )

    test_start_date = (
        series_pct.index[split_index]
    )

    model = arch_model(
        series_pct,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    fitted_model = model.fit(
        last_obs=test_start_date,
        disp="off",
    )

    return fitted_model


# ============================================================
# DIAGNOSTIC TESTS
# ============================================================

def diagnose_model(
    fitted_model,
    asset: str,
) -> dict:
    """
    Diagnose standardized residuals.

    Three questions:

    1. Are residuals autocorrelated?
    2. Are squared residuals autocorrelated?
    3. Is there remaining ARCH structure?
    """

    standardized_residuals = (
        fitted_model
        .std_resid
        .dropna()
    )

    # --------------------------------------------------------
    # Ljung-Box on standardized residuals
    # --------------------------------------------------------

    lb_residuals = acorr_ljungbox(
        standardized_residuals,
        lags=[LAG],
        return_df=True,
    )

    lb_resid_stat = (
        lb_residuals["lb_stat"].iloc[0]
    )

    lb_resid_pvalue = (
        lb_residuals["lb_pvalue"].iloc[0]
    )

    # --------------------------------------------------------
    # Ljung-Box on squared standardized residuals
    # --------------------------------------------------------

    squared_residuals = (
        standardized_residuals ** 2
    )

    lb_squared = acorr_ljungbox(
        squared_residuals,
        lags=[LAG],
        return_df=True,
    )

    lb_squared_stat = (
        lb_squared["lb_stat"].iloc[0]
    )

    lb_squared_pvalue = (
        lb_squared["lb_pvalue"].iloc[0]
    )

    # --------------------------------------------------------
    # ARCH-LM test
    # --------------------------------------------------------

    arch_lm = het_arch(
        standardized_residuals,
        nlags=LAG,
    )

    arch_lm_stat = arch_lm[0]
    arch_lm_pvalue = arch_lm[1]

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    alpha = fitted_model.params.get(
        "alpha[1]",
        float("nan"),
    )

    beta = fitted_model.params.get(
        "beta[1]",
        float("nan"),
    )

    persistence = alpha + beta

    return {
        "asset": asset,
        "observations": len(
            standardized_residuals
        ),
        "lb_resid_stat": lb_resid_stat,
        "lb_resid_pvalue": lb_resid_pvalue,
        "lb_squared_stat": lb_squared_stat,
        "lb_squared_pvalue": lb_squared_pvalue,
        "arch_lm_stat": arch_lm_stat,
        "arch_lm_pvalue": arch_lm_pvalue,
        "persistence_alpha_beta": persistence,
    }


# ============================================================
# RUN ALL ASSETS
# ============================================================

def run_diagnostics(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit and diagnose each asset.
    """

    results = []

    for asset in returns.columns:

        print("\n========================================")
        print(f"DIAGNOSTICS: {asset}")
        print("========================================")

        fitted_model = fit_model(
            returns[asset]
        )

        result = diagnose_model(
            fitted_model,
            asset,
        )

        results.append(result)

        print(
            f"Ljung-Box residuals p-value : "
            f"{result['lb_resid_pvalue']:.6f}"
        )

        print(
            f"Ljung-Box squared residuals : "
            f"{result['lb_squared_pvalue']:.6f}"
        )

        print(
            f"ARCH-LM p-value             : "
            f"{result['arch_lm_pvalue']:.6f}"
        )

    diagnostics = pd.DataFrame(
        results
    )

    return diagnostics


# ============================================================
# INTERPRETATION
# ============================================================

def print_interpretation(
    diagnostics: pd.DataFrame,
) -> None:
    """
    Interpret diagnostic tests.

    For each test:

        p-value >= 0.05
        => we do not detect significant
           remaining structure.

        p-value < 0.05
        => some structure remains.
    """

    print("\n========================================")
    print("GARCH DIAGNOSTIC SUMMARY")
    print("========================================")

    columns = [
        "asset",
        "lb_resid_pvalue",
        "lb_squared_pvalue",
        "arch_lm_pvalue",
        "persistence_alpha_beta",
    ]

    print(
        diagnostics[
            columns
        ].round(6)
    )

    print("\n========================================")
    print("INTERPRETATION")
    print("========================================")

    for _, row in diagnostics.iterrows():

        asset = row["asset"]

        print(f"\n{asset}")
        print("-" * len(asset))

        # Residual autocorrelation

        if row["lb_resid_pvalue"] >= 0.05:
            print(
                "Mean dynamics: OK."
            )
        else:
            print(
                "Mean dynamics: residual "
                "autocorrelation remains."
            )

        # Squared residuals

        if row["lb_squared_pvalue"] >= 0.05:
            print(
                "Volatility dynamics: OK."
            )
        else:
            print(
                "Volatility dynamics: some "
                "dependence remains."
            )

        # ARCH-LM

        if row["arch_lm_pvalue"] >= 0.05:
            print(
                "ARCH-LM: no significant "
                "remaining ARCH effect."
            )
        else:
            print(
                "ARCH-LM: significant "
                "remaining ARCH effect."
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    diagnostics: pd.DataFrame,
) -> None:
    """
    Save diagnostic results.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nDiagnostics saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    returns = load_returns()

    diagnostics = run_diagnostics(
        returns
    )

    save_results(
        diagnostics
    )

    print_interpretation(
        diagnostics
    )

    print(
        "\nGARCH diagnostics completed successfully."
    )


if __name__ == "__main__":
    main()