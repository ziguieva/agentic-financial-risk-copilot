from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import t, chi2


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------

PORTFOLIO_WEIGHTS = {
    "BNP": 1.0 / 3.0,
    "AXA": 1.0 / 3.0,
    "SOCIETE_GENERALE": 1.0 / 3.0,
}


# ------------------------------------------------------------
# Risk parameters
# ------------------------------------------------------------

CORRELATION_WINDOW = 60

CONFIDENCE_LEVELS = [
    0.95,
    0.99,
]


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MODEL_VARIANCE_COLUMNS = {

    "GARCH": "garch_variance",

    "LSTM": "lstm_variance",

    "HYBRID": "hybrid_variance",
}


# ============================================================
# INPUT FILES
# ============================================================

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

FORECASTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
    / "hybrid_forecasts.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "risk"
)

RISK_FORECASTS_FILE = (
    RESULTS_DIR
    / "portfolio_risk_forecasts.csv"
)

BACKTEST_FILE = (
    RESULTS_DIR
    / "var_backtest_summary.csv"
)

CONFIG_FILE = (
    RESULTS_DIR
    / "risk_model_config.json"
)


# ============================================================
# LOAD RETURNS
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load daily logarithmic returns.

    Stored returns are decimal:
        0.01 = 1 %

    We convert them to percentage units
    for consistency with GARCH/LSTM outputs.
    """

    if not RETURNS_FILE.exists():

        raise FileNotFoundError(
            f"Returns file not found: "
            f"{RETURNS_FILE}"
        )

    returns = pd.read_csv(
        RETURNS_FILE,
        index_col=0,
        parse_dates=True,
    )

    returns = (
        returns
        .sort_index()
        .dropna()
    )

    returns_pct = (
        returns * 100.0
    )

    return returns_pct


# ============================================================
# LOAD FORECASTS
# ============================================================

def load_forecasts() -> pd.DataFrame:
    """
    Load GARCH, LSTM and Hybrid variance forecasts.
    """

    if not FORECASTS_FILE.exists():

        raise FileNotFoundError(
            f"Forecast file not found: "
            f"{FORECASTS_FILE}"
        )

    forecasts = pd.read_csv(
        FORECASTS_FILE,
        parse_dates=["date"],
    )

    forecasts = forecasts.sort_values(
        [
            "date",
            "asset",
        ]
    )

    return forecasts


# ============================================================
# CHECK PORTFOLIO
# ============================================================

def validate_portfolio(
    returns: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """
    Validate assets and weights.
    """

    assets = list(
        PORTFOLIO_WEIGHTS.keys()
    )

    weights = np.array(
        [
            PORTFOLIO_WEIGHTS[asset]
            for asset in assets
        ],
        dtype=float,
    )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):

        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    for asset in assets:

        if asset not in returns.columns:

            raise ValueError(
                f"{asset} missing from returns."
            )

        if asset not in forecasts["asset"].unique():

            raise ValueError(
                f"{asset} missing from forecasts."
            )

    print("\n========================================")
    print("PORTFOLIO")
    print("========================================")

    for asset, weight in zip(
        assets,
        weights,
    ):

        print(
            f"{asset:<18} "
            f"{weight * 100:.2f}%"
        )

    return (
        assets,
        weights,
    )


# ============================================================
# TRAINING PORTFOLIO RETURNS
# ============================================================

def build_pretest_portfolio_returns(
    returns: pd.DataFrame,
    assets: list[str],
    weights: np.ndarray,
    test_start_date: pd.Timestamp,
) -> pd.Series:
    """
    Build portfolio returns using ONLY
    observations before the test period.

    Used for Student-t calibration.
    """

    train_returns = returns.loc[
        returns.index < test_start_date,
        assets,
    ]

    portfolio_returns = (
        train_returns.to_numpy()
        @ weights
    )

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=train_returns.index,
        name="portfolio_return_pct",
    )

    return portfolio_returns


# ============================================================
# STUDENT-T CALIBRATION
# ============================================================

def estimate_student_df(
    portfolio_returns: pd.Series,
) -> float:
    """
    Estimate degrees of freedom of a Student-t
    distribution from PRE-TEST portfolio returns.

    Returns are standardized before fitting.

    Only degrees of freedom are retained.

    This parameter controls tail thickness.
    """

    standardized = (

        portfolio_returns
        - portfolio_returns.mean()

    ) / portfolio_returns.std(
        ddof=1
    )

    standardized = (
        standardized
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    degrees_freedom, _, _ = t.fit(
        standardized.to_numpy(),
        floc=0.0,
    )

    # Student-t variance only exists
    # for df > 2.

    degrees_freedom = max(
        float(degrees_freedom),
        2.1,
    )

    print("\n========================================")
    print("STUDENT-T CALIBRATION")
    print("========================================")

    print(
        f"Estimated degrees of freedom: "
        f"{degrees_freedom:.4f}"
    )

    return degrees_freedom


# ============================================================
# ROLLING CORRELATION
# ============================================================

def get_historical_correlation(
    returns: pd.DataFrame,
    assets: list[str],
    date: pd.Timestamp,
) -> np.ndarray:
    """
    Estimate correlation using ONLY information
    available BEFORE the forecast date.

    Window:
        previous 60 observations
    """

    history = returns.loc[
        returns.index < date,
        assets,
    ]

    history = history.tail(
        CORRELATION_WINDOW
    )

    if len(history) < CORRELATION_WINDOW:

        raise RuntimeError(
            f"Not enough history before {date}."
        )

    correlation = (
        history
        .corr()
        .to_numpy()
    )

    return correlation


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def calculate_portfolio_variance(
    individual_variances: np.ndarray,
    correlation: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Build covariance matrix:

        Sigma = D * Corr * D

    where D contains individual volatilities.

    Then:

        portfolio variance =
        w' Sigma w
    """

    individual_variances = np.maximum(
        individual_variances,
        1e-12,
    )

    volatilities = np.sqrt(
        individual_variances
    )

    covariance = (

        np.outer(
            volatilities,
            volatilities,
        )

        * correlation
    )

    portfolio_variance = float(

        weights.T
        @ covariance
        @ weights
    )

    return max(
        portfolio_variance,
        1e-12,
    )


# ============================================================
# STUDENT-T QUANTILE
# ============================================================

def standardized_t_quantile(
    tail_probability: float,
    df: float,
) -> float:
    """
    Quantile of Student-t standardized
    to unit variance.

    Standard t variance:

        df / (df - 2)

    Therefore scaling factor:

        sqrt((df - 2) / df)
    """

    raw_quantile = t.ppf(
        tail_probability,
        df,
    )

    scale = np.sqrt(
        (df - 2.0)
        / df
    )

    return (
        raw_quantile
        * scale
    )


# ============================================================
# VALUE AT RISK
# ============================================================

def calculate_var(
    sigma: float,
    confidence: float,
    df: float,
) -> float:
    """
    One-day Value at Risk.

    Result is expressed as a POSITIVE
    percentage loss.

    Example:
        2.5 means VaR = 2.5% of portfolio value.
    """

    alpha = (
        1.0 - confidence
    )

    quantile = standardized_t_quantile(
        alpha,
        df,
    )

    var = (
        -sigma
        * quantile
    )

    return float(
        max(
            var,
            0.0,
        )
    )


# ============================================================
# EXPECTED SHORTFALL
# ============================================================

def calculate_expected_shortfall(
    sigma: float,
    confidence: float,
    df: float,
) -> float:
    """
    Expected Shortfall for a standardized
    Student-t distribution.

    Result is a POSITIVE percentage loss.
    """

    alpha = (
        1.0 - confidence
    )

    raw_quantile = t.ppf(
        alpha,
        df,
    )

    density = t.pdf(
        raw_quantile,
        df,
    )

    standardization = np.sqrt(
        (df - 2.0)
        / df
    )

    tail_mean = (

        -standardization

        * (
            (df + raw_quantile ** 2)
            / (df - 1.0)
        )

        * density

        / alpha
    )

    es = (
        -sigma
        * tail_mean
    )

    return float(
        max(
            es,
            0.0,
        )
    )


# ============================================================
# BUILD PORTFOLIO RISK FORECASTS
# ============================================================

def build_risk_forecasts(
    returns: pd.DataFrame,
    forecasts: pd.DataFrame,
    assets: list[str],
    weights: np.ndarray,
    df: float,
) -> pd.DataFrame:
    """
    Create daily portfolio risk forecasts
    for:

        GARCH
        LSTM
        HYBRID
    """

    dates = pd.DatetimeIndex(
        sorted(
            forecasts["date"].unique()
        )
    )

    rows = []

    print("\n========================================")
    print("BUILDING RISK FORECASTS")
    print("========================================")

    for model_name, variance_column in (
        MODEL_VARIANCE_COLUMNS.items()
    ):

        print(
            f"Model: {model_name}"
        )

        variance_table = (

            forecasts[
                forecasts["asset"]
                .isin(assets)
            ]

            .pivot(
                index="date",
                columns="asset",
                values=variance_column,
            )

            .reindex(
                columns=assets
            )
        )

        for date in dates:

            if date not in variance_table.index:

                continue

            individual_variances = (
                variance_table
                .loc[date]
                .to_numpy(
                    dtype=float
                )
            )

            if np.isnan(
                individual_variances
            ).any():

                continue

            correlation = (
                get_historical_correlation(
                    returns,
                    assets,
                    date,
                )
            )

            portfolio_variance = (
                calculate_portfolio_variance(
                    individual_variances,
                    correlation,
                    weights,
                )
            )

            sigma_daily_pct = np.sqrt(
                portfolio_variance
            )

            sigma_annualized = (

                sigma_daily_pct
                / 100.0
                * np.sqrt(252)
            )

            # -----------------------------------------------
            # Realized portfolio return
            # -----------------------------------------------

            realized_returns = (

                returns.loc[
                    date,
                    assets,
                ]

                .to_numpy(
                    dtype=float
                )
            )

            portfolio_return = float(

                realized_returns
                @ weights
            )

            row = {

                "date": date,

                "model": model_name,

                "portfolio_return_pct": (
                    portfolio_return
                ),

                "predicted_variance": (
                    portfolio_variance
                ),

                "predicted_volatility_daily_pct": (
                    sigma_daily_pct
                ),

                "predicted_volatility_annualized": (
                    sigma_annualized
                ),
            }

            # -----------------------------------------------
            # VaR / ES
            # -----------------------------------------------

            for confidence in (
                CONFIDENCE_LEVELS
            ):

                label = int(
                    confidence * 100
                )

                row[
                    f"var_{label}_pct"
                ] = calculate_var(
                    sigma_daily_pct,
                    confidence,
                    df,
                )

                row[
                    f"es_{label}_pct"
                ] = (
                    calculate_expected_shortfall(
                        sigma_daily_pct,
                        confidence,
                        df,
                    )
                )

            rows.append(
                row
            )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "model",
            "date",
        ]
    )

    return result


# ============================================================
# KUPIEC TEST
# ============================================================

def kupiec_test(
    violations: int,
    observations: int,
    expected_probability: float,
) -> tuple[float, float]:
    """
    Kupiec unconditional coverage test.

    H0:
        observed VaR violation rate
        equals expected violation probability.

    Returns:
        LR statistic
        p-value
    """

    if observations <= 0:

        return (
            np.nan,
            np.nan,
        )

    observed_probability = (
        violations
        / observations
    )

    epsilon = 1e-12

    p0 = np.clip(
        expected_probability,
        epsilon,
        1.0 - epsilon,
    )

    p1 = np.clip(
        observed_probability,
        epsilon,
        1.0 - epsilon,
    )

    log_likelihood_null = (

        violations
        * np.log(p0)

        +

        (
            observations
            - violations
        )
        * np.log(
            1.0 - p0
        )
    )

    log_likelihood_alt = (

        violations
        * np.log(p1)

        +

        (
            observations
            - violations
        )
        * np.log(
            1.0 - p1
        )
    )

    lr_stat = (

        -2.0
        * (
            log_likelihood_null
            - log_likelihood_alt
        )
    )

    p_value = (

        1.0
        - chi2.cdf(
            lr_stat,
            df=1,
        )
    )

    return (
        float(lr_stat),
        float(p_value),
    )


# ============================================================
# VAR BACKTEST
# ============================================================

def backtest_var(
    risk_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backtest VaR forecasts.

    Violation occurs when:

        realized return < -VaR
    """

    results = []

    for model_name in (
        MODEL_VARIANCE_COLUMNS.keys()
    ):

        model_data = (
            risk_forecasts[
                risk_forecasts[
                    "model"
                ]
                == model_name
            ]
            .copy()
        )

        observations = len(
            model_data
        )

        for confidence in (
            CONFIDENCE_LEVELS
        ):

            label = int(
                confidence * 100
            )

            var_column = (
                f"var_{label}_pct"
            )

            violations = (

                model_data[
                    "portfolio_return_pct"
                ]

                <

                -model_data[
                    var_column
                ]
            )

            violation_count = int(
                violations.sum()
            )

            violation_rate = (

                violation_count
                / observations
            )

            expected_probability = (
                1.0 - confidence
            )

            expected_violations = (

                observations
                * expected_probability
            )

            (
                kupiec_lr,
                kupiec_pvalue,
            ) = kupiec_test(

                violations=violation_count,

                observations=observations,

                expected_probability=(
                    expected_probability
                ),
            )

            results.append(
                {

                    "model": model_name,

                    "confidence": (
                        confidence
                    ),

                    "observations": (
                        observations
                    ),

                    "violations": (
                        violation_count
                    ),

                    "expected_violations": (
                        expected_violations
                    ),

                    "violation_rate": (
                        violation_rate
                    ),

                    "expected_rate": (
                        expected_probability
                    ),

                    "kupiec_lr": (
                        kupiec_lr
                    ),

                    "kupiec_pvalue": (
                        kupiec_pvalue
                    ),

                    "coverage_accepted_5pct": (
                        kupiec_pvalue
                        >= 0.05
                    ),
                }
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_configuration(
    df: float,
) -> None:
    """
    Save risk engine configuration.
    """

    configuration = {

        "portfolio_weights": (
            PORTFOLIO_WEIGHTS
        ),

        "correlation_window": (
            CORRELATION_WINDOW
        ),

        "confidence_levels": (
            CONFIDENCE_LEVELS
        ),

        "distribution": (
            "standardized Student-t"
        ),

        "student_t_degrees_of_freedom": (
            df
        ),

        "market_factors": [
            "CAC40",
            "SP500",
        ],

        "portfolio_assets": list(
            PORTFOLIO_WEIGHTS.keys()
        ),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    risk_forecasts: pd.DataFrame,
    backtest: pd.DataFrame,
    df: float,
) -> None:
    """
    Save risk engine outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_forecasts.to_csv(
        RISK_FORECASTS_FILE,
        index=False,
    )

    backtest.to_csv(
        BACKTEST_FILE,
        index=False,
    )

    save_configuration(
        df
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        RISK_FORECASTS_FILE
    )

    print(
        BACKTEST_FILE
    )

    print(
        CONFIG_FILE
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    risk_forecasts: pd.DataFrame,
    backtest: pd.DataFrame,
) -> None:
    """
    Display key risk results.
    """

    print("\n========================================")
    print("VAR BACKTEST SUMMARY")
    print("========================================")

    display_columns = [

        "model",

        "confidence",

        "observations",

        "violations",

        "expected_violations",

        "violation_rate",

        "kupiec_pvalue",

        "coverage_accepted_5pct",
    ]

    print(

        backtest[
            display_columns
        ]

        .round(6)

        .to_string(
            index=False
        )
    )

    print("\n========================================")
    print("LATEST HYBRID RISK FORECAST")
    print("========================================")

    hybrid = (

        risk_forecasts[
            risk_forecasts["model"]
            == "HYBRID"
        ]

        .sort_values(
            "date"
        )
    )

    latest = (
        hybrid.iloc[-1]
    )

    print(
        f"Date                        : "
        f"{latest['date'].date()}"
    )

    print(
        f"Daily volatility            : "
        f"{latest['predicted_volatility_daily_pct']:.3f}%"
    )

    print(
        f"Annualized volatility       : "
        f"{latest['predicted_volatility_annualized'] * 100:.2f}%"
    )

    print(
        f"VaR 95%                     : "
        f"{latest['var_95_pct']:.3f}%"
    )

    print(
        f"Expected Shortfall 95%      : "
        f"{latest['es_95_pct']:.3f}%"
    )

    print(
        f"VaR 99%                     : "
        f"{latest['var_99_pct']:.3f}%"
    )

    print(
        f"Expected Shortfall 99%      : "
        f"{latest['es_99_pct']:.3f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    returns = load_returns()

    forecasts = load_forecasts()

    assets, weights = (
        validate_portfolio(
            returns,
            forecasts,
        )
    )

    test_start_date = pd.Timestamp(
        forecasts["date"].min()
    )

    print("\n========================================")
    print("RISK TEST PERIOD")
    print("========================================")

    print(
        f"Start : "
        f"{test_start_date.date()}"
    )

    print(
        f"End   : "
        f"{forecasts['date'].max().date()}"
    )

    # ========================================================
    # Student-t calibration using PRE-TEST data only
    # ========================================================

    pretest_portfolio_returns = (
        build_pretest_portfolio_returns(

            returns,

            assets,

            weights,

            test_start_date,
        )
    )

    degrees_freedom = (
        estimate_student_df(
            pretest_portfolio_returns
        )
    )

    # ========================================================
    # Daily risk forecasts
    # ========================================================

    risk_forecasts = (
        build_risk_forecasts(

            returns,

            forecasts,

            assets,

            weights,

            degrees_freedom,
        )
    )

    # ========================================================
    # VaR backtesting
    # ========================================================

    backtest = backtest_var(
        risk_forecasts
    )

    # ========================================================
    # Save
    # ========================================================

    save_results(

        risk_forecasts,

        backtest,

        degrees_freedom,
    )

    # ========================================================
    # Summary
    # ========================================================

    print_summary(

        risk_forecasts,

        backtest,
    )

    print(
        "\nPortfolio risk engine "
        "completed successfully."
    )


if __name__ == "__main__":
    main()