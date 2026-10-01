from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20

MAX_EPOCHS = 120
EARLY_STOPPING_PATIENCE = 15

GRADIENT_CLIP = 1.0


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "lstm"
)

MODEL_FILE = (
    RESULTS_DIR
    / "lstm_volatility.pt"
)

HISTORY_FILE = (
    RESULTS_DIR
    / "training_history.csv"
)

FORECAST_FILE = (
    RESULTS_DIR
    / "lstm_forecasts.csv"
)

METRICS_FILE = (
    RESULTS_DIR
    / "lstm_metrics.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed() -> None:
    """
    Make training as reproducible as possible.
    """

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    """
    Select best available device.

    Priority:
        Apple MPS
        CUDA
        CPU
    """

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    return device


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load prepared LSTM arrays and metadata.
    """

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
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
        "X_train": data["X_train"],
        "y_train": data["y_train"],

        "X_val": data["X_val"],
        "y_val": data["y_val"],

        "X_test": data["X_test"],
        "y_test": data["y_test"],

        "y_test_raw": data["y_test_raw"],

        "test_target_dates": data[
            "test_target_dates"
        ],
    }

    print("\nDataset loaded.")

    print(
        f"X_train : "
        f"{dataset['X_train'].shape}"
    )

    print(
        f"X_val   : "
        f"{dataset['X_val'].shape}"
    )

    print(
        f"X_test  : "
        f"{dataset['X_test'].shape}"
    )

    return dataset, metadata


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    dataset: dict,
):
    """
    Convert numpy arrays into PyTorch DataLoaders.
    """

    X_train = torch.tensor(
        dataset["X_train"],
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        dataset["y_train"],
        dtype=torch.float32,
    )

    X_val = torch.tensor(
        dataset["X_val"],
        dtype=torch.float32,
    )

    y_val = torch.tensor(
        dataset["y_val"],
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    val_dataset = TensorDataset(
        X_val,
        y_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# MODEL
# ============================================================

class VolatilityLSTM(nn.Module):
    """
    Multivariate LSTM for next-day variance forecasting.

    Input:
        batch x 30 x 25

    Output:
        batch x 5

    The model predicts:

        log(1 + variance)

    for all assets simultaneously.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )

        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                32,
                output_size,
            ),

            # log1p(variance) must be >= 0
            nn.Softplus(),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.lstm(x)

        # Last time step
        last_hidden = output[:, -1, :]

        prediction = self.head(
            last_hidden
        )

        return prediction


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
) -> float:
    """
    Train model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_loss(
    model,
    loader,
    criterion,
    device,
) -> float:
    """
    Compute validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
):
    """
    Train model with early stopping.
    """

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []

    print("\n========================================")
    print("TRAINING")
    print("========================================")

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping activated."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    print("\n========================================")
    print("BEST MODEL")
    print("========================================")

    print(
        f"Best epoch    : {best_epoch}"
    )

    print(
        f"Best val loss : "
        f"{best_val_loss:.6f}"
    )

    history_df = pd.DataFrame(
        history
    )

    return (
        model,
        history_df,
        best_epoch,
        best_val_loss,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_test(
    model,
    X_test: np.ndarray,
    device,
) -> np.ndarray:
    """
    Predict test set.

    Output is converted from:

        log(1 + variance)

    back to:

        variance
    """

    model.eval()

    tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        predictions_log = model(
            tensor
        )

    predictions_log = (
        predictions_log
        .cpu()
        .numpy()
    )

    predictions_variance = np.expm1(
        predictions_log
    )

    predictions_variance = np.maximum(
        predictions_variance,
        1e-8,
    )

    return predictions_variance


# ============================================================
# QLIKE
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
        np.log(predicted)
        +
        actual / predicted
    )

    return float(
        np.mean(loss)
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    actual_variance: np.ndarray,
    predicted_variance: np.ndarray,
    assets: list[str],
) -> pd.DataFrame:
    """
    Compute test metrics for each asset.
    """

    results = []

    for i, asset in enumerate(assets):

        actual = actual_variance[:, i]

        predicted = (
            predicted_variance[:, i]
        )

        mse = np.mean(
            (actual - predicted) ** 2
        )

        mae = np.mean(
            np.abs(
                actual - predicted
            )
        )

        qlike = qlike_loss(
            actual,
            predicted,
        )

        if (
            np.std(actual) > 0
            and np.std(predicted) > 0
        ):

            correlation = np.corrcoef(
                actual,
                predicted,
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
# FORECAST DATAFRAME
# ============================================================

def create_forecast_dataframe(
    dates,
    actual_variance,
    predicted_variance,
    assets,
) -> pd.DataFrame:
    """
    Create long-format forecast table.
    """

    rows = []

    for date_index, date in enumerate(
        dates
    ):

        for asset_index, asset in enumerate(
            assets
        ):

            actual = float(
                actual_variance[
                    date_index,
                    asset_index,
                ]
            )

            predicted = float(
                predicted_variance[
                    date_index,
                    asset_index,
                ]
            )

            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "realized_variance": actual,
                    "predicted_variance": predicted,
                    "predicted_volatility_daily_pct": (
                        np.sqrt(predicted)
                    ),
                    "predicted_volatility_annualized": (
                        np.sqrt(predicted)
                        / 100.0
                        * np.sqrt(252)
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
    model,
    history,
    forecasts,
    metrics,
    metadata,
    input_size,
    output_size,
    best_epoch,
    best_val_loss,
) -> None:
    """
    Save model and experiment outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),

        "input_size": input_size,

        "output_size": output_size,

        "hidden_size": HIDDEN_SIZE,

        "num_layers": NUM_LAYERS,

        "dropout": DROPOUT,

        "sequence_length": metadata[
            "sequence_length"
        ],

        "target_names": metadata[
            "target_names"
        ],

        "best_epoch": best_epoch,

        "best_val_loss": best_val_loss,
    }

    torch.save(
        checkpoint,
        MODEL_FILE,
    )

    history.to_csv(
        HISTORY_FILE,
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

    print(MODEL_FILE)
    print(HISTORY_FILE)
    print(FORECAST_FILE)
    print(METRICS_FILE)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed()

    device = get_device()

    dataset, metadata = (
        load_dataset()
    )

    (
        train_loader,
        val_loader,
    ) = create_dataloaders(
        dataset
    )

    input_size = (
        dataset["X_train"].shape[2]
    )

    output_size = (
        dataset["y_train"].shape[1]
    )

    model = VolatilityLSTM(
        input_size=input_size,
        output_size=output_size,
    )

    model = model.to(
        device
    )

    print("\n========================================")
    print("MODEL")
    print("========================================")

    print(model)

    number_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{number_parameters:,}"
    )

    (
        model,
        history,
        best_epoch,
        best_val_loss,
    ) = train_model(
        model,
        train_loader,
        val_loader,
        device,
    )

    predicted_variance = predict_test(
        model,
        dataset["X_test"],
        device,
    )

    actual_variance = (
        dataset["y_test_raw"]
    )

    assets = metadata[
        "target_names"
    ]

    metrics = compute_metrics(
        actual_variance,
        predicted_variance,
        assets,
    )

    forecasts = (
        create_forecast_dataframe(
            dataset[
                "test_target_dates"
            ],
            actual_variance,
            predicted_variance,
            assets,
        )
    )

    save_results(
        model,
        history,
        forecasts,
        metrics,
        metadata,
        input_size,
        output_size,
        best_epoch,
        best_val_loss,
    )

    print("\n========================================")
    print("LSTM TEST SUMMARY")
    print("========================================")

    print(
        metrics.round(6)
    )

    print(
        "\nLSTM volatility model "
        "completed successfully."
    )


if __name__ == "__main__":
    main()