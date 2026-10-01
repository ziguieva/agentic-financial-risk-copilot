from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from arch import arch_model


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHT_GRID = np.linspace(
    0.0,
    1.0,
    101,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# INPUT FILES
# ============================================================

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_dataset.npz"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_dataset_metadata.json"
)

LSTM_MODEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "lstm"
    / "lstm_volatility.pt"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
)

WEIGHTS_FILE = (
    RESULTS_DIR
    / "hybrid_weights.csv"
)

FORECAST_FILE = (
    RESULTS_DIR
    / "hybrid_forecasts.csv"
)

METRICS_FILE = (
    RESULTS_DIR
    / "hybrid_metrics.csv"
)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    """
    Select best available computing device.
    """

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD RETURNS
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load historical log returns.
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

    print("Returns loaded.")

    print(
        f"Observations : {len(returns)}"
    )

    print(
        f"Assets       : {len(returns.columns)}"
    )

    return returns


# ============================================================
# LOAD LSTM DATASET
# ============================================================

def load_lstm_data() -> tuple:
    """
    Load validation and test datasets.
    """

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"LSTM dataset not found: {DATASET_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    data = np.load(
        DATASET_FILE,
        allow_pickle=False,
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    dataset = {

        "X_val": data[
            "X_val"
        ],

        "y_val_raw": data[
            "y_val_raw"
        ],

        "X_test": data[
            "X_test"
        ],

        "y_test_raw": data[
            "y_test_raw"
        ],

        "val_dates": pd.to_datetime(
            data[
                "val_target_dates"
            ]
        ),

        "test_dates": pd.to_datetime(
            data[
                "test_target_dates"
            ]
        ),
    }

    return (
        dataset,
        metadata,
    )


# ============================================================
# LSTM MODEL
# ============================================================

class VolatilityLSTM(nn.Module):
    """
    Architecture matching the previously
    trained LSTM model.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                output_size,
            ),

            nn.Softplus(),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.lstm(
            x
        )

        last_hidden = (
            output[:, -1, :]
        )

        return self.head(
            last_hidden
        )


# ============================================================
# LOAD TRAINED LSTM
# ============================================================

def load_lstm_model(
    device: torch.device,
):
    """
    Restore best trained LSTM model.
    """

    if not LSTM_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {LSTM_MODEL_FILE}"
        )

    checkpoint = torch.load(
        LSTM_MODEL_FILE,
        map_location=device,
        weights_only=True,
    )

    model = VolatilityLSTM(

        input_size=checkpoint[
            "input_size"
        ],

        output_size=checkpoint[
            "output_size"
        ],

        hidden_size=checkpoint[
            "hidden_size"
        ],

        num_layers=checkpoint[
            "num_layers"
        ],

        dropout=checkpoint[
            "dropout"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "LSTM model loaded successfully."
    )

    return model


# ============================================================
# LSTM PREDICTION
# ============================================================

def predict_lstm(
    model,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Predict realized variance.

    Network output:
        log(1 + variance)

    Convert back:
        variance = exp(output) - 1
    """

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(
        device
    )

    with torch.no_grad():

        prediction_log = model(
            tensor
        )

    prediction_log = (
        prediction_log
        .cpu()
        .numpy()
    )

    prediction_variance = (
        np.expm1(
            prediction_log
        )
    )

    return np.maximum(
        prediction_variance,
        1e-8,
    )


# ============================================================
# GARCH FORECAST
# ============================================================

def generate_garch_forecasts(
    series: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Fit AR(1)-GARCH(1,1)-Student-t using
    only observations before start_date.

    Then generate one-step-ahead
    conditional variance forecasts
    throughout the requested period.

    Returns are expressed in percentage form,
    exactly as in the original GARCH baseline.
    """

    series_pct = (
        series * 100.0
    )

    start_position = (
        series_pct.index.get_loc(
            start_date
        )
    )

    if start_position == 0:
        raise ValueError(
            "Forecast period starts too early."
        )

    forecast_start_date = (
        series_pct.index[
            start_position - 1
        ]
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

        last_obs=start_date,

        disp="off",
    )

    forecasts = (
        fitted_model.forecast(

            horizon=1,

            start=forecast_start_date,

            align="target",

            reindex=True,
        )
    )

    variance = (

        forecasts
        .variance
        .iloc[:, 0]
        .loc[
            start_date:end_date
        ]
    )

    variance = (
        variance.dropna()
    )

    return variance


# ============================================================
# ALL GARCH FORECASTS
# ============================================================

def build_garch_predictions(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    assets: list[str],
) -> np.ndarray:
    """
    Build GARCH predictions for all assets
    over a specific date window.
    """

    start_date = dates[0]
    end_date = dates[-1]

    predictions = []

    for asset in assets:

        print(
            f"GARCH forecasting: {asset}"
        )

        variance = (
            generate_garch_forecasts(

                returns[asset],

                start_date=start_date,

                end_date=end_date,
            )
        )

        variance = variance.reindex(
            dates
        )

        if variance.isna().any():

            missing = (
                variance[
                    variance.isna()
                ].index
            )

            raise RuntimeError(
                f"Missing GARCH forecasts "
                f"for {asset}: {missing}"
            )

        predictions.append(
            variance.to_numpy()
        )

    predictions = np.column_stack(
        predictions
    )

    return predictions


# ============================================================
# QLIKE LOSS
# ============================================================

def qlike_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    QLIKE volatility loss.

    Lower is better.
    """

    epsilon = 1e-12

    predicted = np.maximum(
        predicted,
        epsilon,
    )

    loss = (
        np.log(
            predicted
        )
        +
        actual / predicted
    )

    return float(
        np.mean(
            loss
        )
    )


# ============================================================
# OPTIMIZE HYBRID WEIGHT
# ============================================================

def find_best_weight(
    actual: np.ndarray,
    garch_prediction: np.ndarray,
    lstm_prediction: np.ndarray,
) -> tuple[float, float]:
    """
    Search for best LSTM weight
    using VALIDATION DATA ONLY.

    Hybrid:

        (1-w) * GARCH
        +
        w * LSTM

    w = 0:
        pure GARCH

    w = 1:
        pure LSTM
    """

    best_weight = None

    best_loss = float(
        "inf"
    )

    for weight in WEIGHT_GRID:

        hybrid = (

            (1.0 - weight)
            * garch_prediction

            +

            weight
            * lstm_prediction
        )

        loss = qlike_loss(
            actual,
            hybrid,
        )

        if loss < best_loss:

            best_loss = loss

            best_weight = float(
                weight
            )

    return (
        best_weight,
        best_loss,
    )


# ============================================================
# LEARN WEIGHTS
# ============================================================

def learn_hybrid_weights(
    actual_val: np.ndarray,
    garch_val: np.ndarray,
    lstm_val: np.ndarray,
    assets: list[str],
) -> pd.DataFrame:
    """
    Learn one validation-based
    ensemble weight per asset.
    """

    results = []

    print("\n========================================")
    print("VALIDATION WEIGHT OPTIMIZATION")
    print("========================================")

    for i, asset in enumerate(
        assets
    ):

        actual = (
            actual_val[:, i]
        )

        garch = (
            garch_val[:, i]
        )

        lstm = (
            lstm_val[:, i]
        )

        weight, hybrid_qlike = (
            find_best_weight(

                actual,

                garch,

                lstm,
            )
        )

        garch_qlike = qlike_loss(
            actual,
            garch,
        )

        lstm_qlike = qlike_loss(
            actual,
            lstm,
        )

        results.append(
            {

                "asset": asset,

                "lstm_weight": (
                    weight
                ),

                "garch_weight": (
                    1.0 - weight
                ),

                "validation_garch_qlike": (
                    garch_qlike
                ),

                "validation_lstm_qlike": (
                    lstm_qlike
                ),

                "validation_hybrid_qlike": (
                    hybrid_qlike
                ),
            }
        )

        print(
            f"{asset:<18} | "
            f"GARCH: {1-weight:.2f} | "
            f"LSTM: {weight:.2f} | "
            f"Hybrid QLIKE: "
            f"{hybrid_qlike:.6f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# BUILD HYBRID TEST PREDICTIONS
# ============================================================

def create_hybrid_predictions(
    weights: pd.DataFrame,
    garch_test: np.ndarray,
    lstm_test: np.ndarray,
    assets: list[str],
) -> np.ndarray:
    """
    Apply validation-selected weights
    to unseen test predictions.
    """

    hybrid = np.zeros_like(
        garch_test
    )

    for i, asset in enumerate(
        assets
    ):

        row = weights.loc[
            weights["asset"]
            == asset
        ].iloc[0]

        lstm_weight = float(
            row[
                "lstm_weight"
            ]
        )

        garch_weight = (
            1.0
            - lstm_weight
        )

        hybrid[:, i] = (

            garch_weight
            * garch_test[:, i]

            +

            lstm_weight
            * lstm_test[:, i]
        )

    return hybrid


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    assets: list[str],
) -> pd.DataFrame:
    """
    Compute evaluation metrics.
    """

    results = []

    for i, asset in enumerate(
        assets
    ):

        y = actual[:, i]

        y_hat = predicted[:, i]

        mse = np.mean(
            (y - y_hat) ** 2
        )

        mae = np.mean(
            np.abs(
                y - y_hat
            )
        )

        qlike = qlike_loss(
            y,
            y_hat,
        )

        if (
            np.std(y) > 0
            and
            np.std(y_hat) > 0
        ):

            correlation = np.corrcoef(
                y,
                y_hat,
            )[0, 1]

        else:

            correlation = np.nan

        results.append(
            {

                "asset": asset,

                "mse": mse,

                "mae": mae,

                "qlike": qlike,

                "correlation": correlation,
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# FORECAST TABLE
# ============================================================

def create_forecast_dataframe(
    dates: pd.DatetimeIndex,
    assets: list[str],
    actual: np.ndarray,
    garch: np.ndarray,
    lstm: np.ndarray,
    hybrid: np.ndarray,
) -> pd.DataFrame:
    """
    Save all three forecasts
    in long format.
    """

    rows = []

    for date_index, date in enumerate(
        dates
    ):

        for asset_index, asset in enumerate(
            assets
        ):

            rows.append(
                {

                    "date": (
                        date.strftime(
                            "%Y-%m-%d"
                        )
                    ),

                    "asset": asset,

                    "realized_variance": float(
                        actual[
                            date_index,
                            asset_index,
                        ]
                    ),

                    "garch_variance": float(
                        garch[
                            date_index,
                            asset_index,
                        ]
                    ),

                    "lstm_variance": float(
                        lstm[
                            date_index,
                            asset_index,
                        ]
                    ),

                    "hybrid_variance": float(
                        hybrid[
                            date_index,
                            asset_index,
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    weights: pd.DataFrame,
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """
    Save hybrid experiment outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights.to_csv(
        WEIGHTS_FILE,
        index=False,
    )

    forecasts.to_csv(
        FORECAST_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        WEIGHTS_FILE
    )

    print(
        FORECAST_FILE
    )

    print(
        METRICS_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    device = get_device()

    print(
        f"Device: {device}"
    )

    returns = load_returns()

    (
        dataset,
        metadata,
    ) = load_lstm_data()

    assets = metadata[
        "target_names"
    ]

    model = load_lstm_model(
        device
    )

    # ========================================================
    # LSTM VALIDATION
    # ========================================================

    print("\nGenerating LSTM validation forecasts...")

    lstm_val = predict_lstm(
        model,
        dataset[
            "X_val"
        ],
        device,
    )

    # ========================================================
    # LSTM TEST
    # ========================================================

    print(
        "Generating LSTM test forecasts..."
    )

    lstm_test = predict_lstm(
        model,
        dataset[
            "X_test"
        ],
        device,
    )

    # ========================================================
    # GARCH VALIDATION
    # ========================================================

    print(
        "\nGenerating GARCH validation forecasts..."
    )

    garch_val = build_garch_predictions(

        returns,

        dataset[
            "val_dates"
        ],

        assets,
    )

    # ========================================================
    # LEARN WEIGHTS ON VALIDATION
    # ========================================================

    weights = learn_hybrid_weights(

        actual_val=dataset[
            "y_val_raw"
        ],

        garch_val=garch_val,

        lstm_val=lstm_val,

        assets=assets,
    )

    # ========================================================
    # GARCH TEST
    # ========================================================

    print(
        "\nGenerating GARCH test forecasts..."
    )

    garch_test = build_garch_predictions(

        returns,

        dataset[
            "test_dates"
        ],

        assets,
    )

    # ========================================================
    # HYBRID TEST
    # ========================================================

    hybrid_test = (
        create_hybrid_predictions(

            weights,

            garch_test,

            lstm_test,

            assets,
        )
    )

    # ========================================================
    # TEST METRICS
    # ========================================================

    metrics = compute_metrics(

        actual=dataset[
            "y_test_raw"
        ],

        predicted=hybrid_test,

        assets=assets,
    )

    # ========================================================
    # FORECAST TABLE
    # ========================================================

    forecasts = (
        create_forecast_dataframe(

            dates=dataset[
                "test_dates"
            ],

            assets=assets,

            actual=dataset[
                "y_test_raw"
            ],

            garch=garch_test,

            lstm=lstm_test,

            hybrid=hybrid_test,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(

        weights,

        forecasts,

        metrics,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n========================================")
    print("HYBRID TEST SUMMARY")
    print("========================================")

    print(
        metrics.round(
            6
        )
    )

    print(
        "\nHybrid GARCH-LSTM "
        "completed successfully."
    )


if __name__ == "__main__":
    main()