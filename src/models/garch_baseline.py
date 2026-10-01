from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "garch"
)

FORECAST_FILE = (
    RESULTS_DIR
    / "garch_forecasts.csv"
)

METRICS_FILE = (
    RESULTS_DIR
    / "garch_metrics.csv"
)

PARAMETERS_FILE = (
    RESULTS_DIR
    / "garch_parameters.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load logarithmic returns.

    Returns are stored in decimal form:
        0.01 = 1 %

    For GARCH estimation, we convert them to percentage form
    because numerical optimization is generally more stable.
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
# QLIKE LOSS
# ============================================================

def qlike_loss(
    actual_variance: np.ndarray,
    predicted_variance: np.ndarray,
) -> float:
    """
    QLIKE loss for volatility forecasts.

    Lower is better.

    It is widely useful for evaluating variance forecasts.

    Formula:
        log(sigma_hat^2)
        +
        realized_variance / sigma_hat^2
    """

    epsilon = 1e-12

    predicted_variance = np.maximum(
        predicted_variance,
        epsilon,
    )

    loss = (
        np.log(predicted_variance)
        +
        actual_variance / predicted_variance
    )

    return float(
        np.mean(loss)
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    actual_variance: pd.Series,
    predicted_variance: pd.Series,
) -> dict:
    """
    Compute volatility forecasting metrics.

    The realized variance proxy is:
        r_t^2

    The model prediction is:
        sigma_t^2
    """

    actual = actual_variance.to_numpy()

    predicted = predicted_variance.to_numpy()

    mse = np.mean(
        (actual - predicted) ** 2
    )

    mae = np.mean(
        np.abs(actual - predicted)
    )

    qlike = qlike_loss(
        actual,
        predicted,
    )

    correlation = np.corrcoef(
        actual,
        predicted,
    )[0, 1]

    return {
        "mse": mse,
        "mae": mae,
        "qlike": qlike,
        "correlation": correlation,
    }


# ============================================================
# GARCH MODEL
# ============================================================

def fit_garch_model(
    series: pd.Series,
    asset: str,
) -> tuple:
    """
    Fit an AR(1)-GARCH(1,1) model
    with Student-t innovations.

    Parameters are estimated ONLY
    using the training period.

    Forecasts are then generated
    over the test period.
    """

    print("\n========================================")
    print(f"GARCH MODEL: {asset}")
    print("========================================")

    # --------------------------------------------------------
    # Convert returns to percentage scale
    # --------------------------------------------------------

    series_pct = series * 100.0

    # --------------------------------------------------------
    # Temporal train/test split
    # --------------------------------------------------------

    split_index = int(
        len(series_pct) * TRAIN_RATIO
    )

    if split_index <= 1:
        raise ValueError(
            f"Not enough observations for {asset}"
        )

    test_start_date = (
        series_pct.index[split_index]
    )

    forecast_start_date = (
        series_pct.index[split_index - 1]
    )

    train_size = split_index

    test_size = (
        len(series_pct) - split_index
    )

    print(
        f"Train observations : {train_size}"
    )

    print(
        f"Test observations  : {test_size}"
    )

    print(
        f"Train end          : "
        f"{series_pct.index[split_index - 1].date()}"
    )

    print(
        f"Test start         : "
        f"{test_start_date.date()}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    # last_obs ensures parameters are estimated
    # only using observations before test_start_date.

    fitted_model = model.fit(
        last_obs=test_start_date,
        disp="off",
    )

    # --------------------------------------------------------
    # One-step-ahead forecasts
    # --------------------------------------------------------

    forecasts = fitted_model.forecast(
        horizon=1,
        start=forecast_start_date,
        align="target",
        reindex=True,
    )

    predicted_variance = (
        forecasts
        .variance
        .iloc[:, 0]
        .loc[test_start_date:]
        .dropna()
    )

    predicted_variance.name = (
        "predicted_variance"
    )

    # --------------------------------------------------------
    # Realized variance proxy
    #
    # r_t^2
    # --------------------------------------------------------

    actual_returns = (
        series_pct.loc[
            predicted_variance.index
        ]
    )

    realized_variance = (
        actual_returns ** 2
    )

    realized_variance.name = (
        "realized_variance"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_metrics(
        actual_variance=realized_variance,
        predicted_variance=predicted_variance,
    )

    # --------------------------------------------------------
    # Model persistence
    #
    # alpha + beta close to 1
    # = persistent volatility
    # --------------------------------------------------------

    alpha = fitted_model.params.get(
        "alpha[1]",
        np.nan,
    )

    beta = fitted_model.params.get(
        "beta[1]",
        np.nan,
    )

    persistence = alpha + beta

    print("\nEstimated parameters:")

    print(
        fitted_model.params.round(6)
    )

    print(
        f"\nVolatility persistence "
        f"(alpha + beta): "
        f"{persistence:.6f}"
    )

    print("\nTest metrics:")

    print(
        f"MSE         : "
        f"{metrics['mse']:.6f}"
    )

    print(
        f"MAE         : "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"QLIKE       : "
        f"{metrics['qlike']:.6f}"
    )

    print(
        f"Correlation : "
        f"{metrics['correlation']:.6f}"
    )

    # --------------------------------------------------------
    # Forecast dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "actual_return_pct": actual_returns,
            "realized_variance": realized_variance,
            "predicted_variance": predicted_variance,
        }
    )

    result[
        "predicted_volatility_daily_pct"
    ] = np.sqrt(
        result["predicted_variance"]
    )

    result[
        "predicted_volatility_annualized"
    ] = (
        np.sqrt(
            result["predicted_variance"]
        )
        / 100.0
        * np.sqrt(252)
    )

    return (
        fitted_model,
        result,
        metrics,
        persistence,
        train_size,
        test_size,
    )


# ============================================================
# RUN ALL ASSETS
# ============================================================

def run_all_models(
    returns: pd.DataFrame,
) -> tuple:
    """
    Fit one GARCH model per asset.
    """

    all_forecasts = []

    metrics_results = []

    parameters_results = []

    for asset in returns.columns:

        (
            model,
            forecasts,
            metrics,
            persistence,
            train_size,
            test_size,
        ) = fit_garch_model(
            returns[asset],
            asset,
        )

        # ----------------------------------------------------
        # Forecasts
        # ----------------------------------------------------

        forecasts = forecasts.copy()

        forecasts["asset"] = asset

        forecasts = (
            forecasts
            .reset_index()
            .rename(
                columns={
                    "index": "date"
                }
            )
        )

        all_forecasts.append(
            forecasts
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics_results.append(
            {
                "asset": asset,
                "train_size": train_size,
                "test_size": test_size,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "qlike": metrics["qlike"],
                "correlation": metrics[
                    "correlation"
                ],
                "persistence_alpha_beta": (
                    persistence
                ),
            }
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        parameters = {
            "asset": asset,
        }

        for (
            parameter_name,
            parameter_value,
        ) in model.params.items():

            parameters[
                parameter_name
            ] = parameter_value

        parameters_results.append(
            parameters
        )

    forecasts_df = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        metrics_results
    )

    parameters_df = pd.DataFrame(
        parameters_results
    )

    return (
        forecasts_df,
        metrics_df,
        parameters_df,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    """
    Save GARCH outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        FORECAST_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    parameters.to_csv(
        PARAMETERS_FILE,
        index=False,
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(FORECAST_FILE)
    print(METRICS_FILE)
    print(PARAMETERS_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    metrics: pd.DataFrame,
) -> None:
    """
    Display concise final comparison.
    """

    print("\n========================================")
    print("GARCH BASELINE SUMMARY")
    print("========================================")

    display_columns = [
        "asset",
        "qlike",
        "correlation",
        "persistence_alpha_beta",
    ]

    print(
        metrics[
            display_columns
        ].round(6)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    returns = load_returns()

    (
        forecasts,
        metrics,
        parameters,
    ) = run_all_models(
        returns
    )

    save_results(
        forecasts,
        metrics,
        parameters,
    )

    print_final_summary(
        metrics
    )

    print(
        "\nGARCH baseline completed successfully."
    )


if __name__ == "__main__":
    main()