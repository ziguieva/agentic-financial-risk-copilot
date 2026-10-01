from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30

FINAL_TEST_RATIO = 0.20

# Among the remaining 80%:
# 87.5% train + 12.5% validation
# gives approximately:
#
# 70% train
# 10% validation
# 20% test

TRAIN_WITHIN_PRETEST_RATIO = 0.875


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_returns.csv"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_dataset.npz"
)

OUTPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_dataset_metadata.json"
)


# ============================================================
# LOAD RETURNS
# ============================================================

def load_returns() -> pd.DataFrame:
    """
    Load logarithmic returns.
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
# BUILD FEATURES
# ============================================================

def build_features(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build multivariate financial features.

    Returns are converted from decimal form:

        0.01

    to percentage form:

        1.0

    This keeps the scale consistent
    with the GARCH implementation.
    """

    returns_pct = returns * 100.0

    features = pd.DataFrame(
        index=returns.index
    )

    for asset in returns.columns:

        r = returns_pct[asset]

        # Daily return
        features[
            f"{asset}_return"
        ] = r

        # Absolute return
        features[
            f"{asset}_abs_return"
        ] = r.abs()

        # Squared return
        features[
            f"{asset}_squared_return"
        ] = r ** 2

        # Short-term volatility
        features[
            f"{asset}_volatility_5"
        ] = r.rolling(
            window=5
        ).std()

        # Medium-term volatility
        features[
            f"{asset}_volatility_20"
        ] = r.rolling(
            window=20
        ).std()

    features = features.dropna()

    print("\nFeature engineering completed.")

    print(
        f"Features per day : "
        f"{len(features.columns)}"
    )

    print(
        f"Usable rows      : "
        f"{len(features)}"
    )

    return features


# ============================================================
# BUILD TARGETS
# ============================================================

def build_targets(
    returns: pd.DataFrame,
    feature_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Target:

        next-day realized variance

    Proxy:

        r_(t+1)^2

    Information up to day t is used
    to predict variance at t+1.
    """

    returns_pct = returns * 100.0

    next_day_variance = (
        returns_pct.shift(-1) ** 2
    )

    next_day_variance = (
        next_day_variance
        .reindex(feature_index)
    )

    # --------------------------------------------------------
    # Map current date -> next trading date
    # --------------------------------------------------------

    current_dates = returns.index[:-1]
    following_dates = returns.index[1:]

    date_mapping = pd.Series(
        following_dates,
        index=current_dates,
    )

    target_dates = (
        date_mapping
        .reindex(feature_index)
    )

    # --------------------------------------------------------
    # Remove rows without a next-day target
    # --------------------------------------------------------

    valid_mask = (
        next_day_variance
        .notna()
        .all(axis=1)
        &
        target_dates.notna()
    )

    next_day_variance = (
        next_day_variance.loc[
            valid_mask
        ]
    )

    target_dates = (
        target_dates.loc[
            valid_mask
        ]
    )

    return (
        next_day_variance,
        target_dates,
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_dates: pd.Series,
) -> tuple:
    """
    Convert time-series into LSTM sequences.

    Input:
        30 days x 25 features

    Output:
        next-day variance for 5 assets
    """

    common_index = (
        features.index
        .intersection(targets.index)
        .intersection(target_dates.index)
    )

    features = features.loc[
        common_index
    ]

    targets = targets.loc[
        common_index
    ]

    target_dates = target_dates.loc[
        common_index
    ]

    X = []
    y = []

    sequence_end_dates = []
    sequence_target_dates = []

    for i in range(
        SEQUENCE_LENGTH - 1,
        len(features),
    ):

        start = (
            i - SEQUENCE_LENGTH + 1
        )

        end = i + 1

        sequence = (
            features
            .iloc[start:end]
            .to_numpy(
                dtype=np.float32
            )
        )

        target = (
            targets
            .iloc[i]
            .to_numpy(
                dtype=np.float32
            )
        )

        X.append(sequence)
        y.append(target)

        sequence_end_dates.append(
            features.index[i]
        )

        sequence_target_dates.append(
            target_dates.iloc[i]
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    sequence_end_dates = pd.DatetimeIndex(
        sequence_end_dates
    )

    sequence_target_dates = pd.DatetimeIndex(
        sequence_target_dates
    )

    return (
        X,
        y,
        sequence_end_dates,
        sequence_target_dates,
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    end_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    original_returns: pd.DataFrame,
) -> dict:
    """
    Temporal split:

        Train      ~70%
        Validation ~10%
        Test       ~20%

    Final test period matches
    the GARCH baseline.

    No random shuffle.
    """

    garch_split_index = int(
        len(original_returns)
        * (1.0 - FINAL_TEST_RATIO)
    )

    test_start_date = (
        original_returns
        .index[garch_split_index]
    )

    print("\n========================================")
    print("TEMPORAL SPLIT")
    print("========================================")

    print(
        f"Final test starts : "
        f"{test_start_date.date()}"
    )

    # --------------------------------------------------------
    # Split according to TARGET date
    # --------------------------------------------------------

    pretest_mask = (
        target_dates < test_start_date
    )

    test_mask = (
        target_dates >= test_start_date
    )

    pretest_indices = np.where(
        pretest_mask
    )[0]

    test_indices = np.where(
        test_mask
    )[0]

    if len(pretest_indices) == 0:
        raise RuntimeError(
            "No pre-test observations available."
        )

    if len(test_indices) == 0:
        raise RuntimeError(
            "No test observations available."
        )

    # --------------------------------------------------------
    # Split pre-test into train / validation
    # --------------------------------------------------------

    train_count = int(
        len(pretest_indices)
        * TRAIN_WITHIN_PRETEST_RATIO
    )

    train_indices = (
        pretest_indices[
            :train_count
        ]
    )

    validation_indices = (
        pretest_indices[
            train_count:
        ]
    )

    split = {

        "X_train": X[
            train_indices
        ],

        "y_train": y[
            train_indices
        ],

        "X_val": X[
            validation_indices
        ],

        "y_val": y[
            validation_indices
        ],

        "X_test": X[
            test_indices
        ],

        "y_test": y[
            test_indices
        ],

        "train_target_dates": target_dates[
            train_indices
        ],

        "val_target_dates": target_dates[
            validation_indices
        ],

        "test_target_dates": target_dates[
            test_indices
        ],

        "train_end_dates": end_dates[
            train_indices
        ],

        "val_end_dates": end_dates[
            validation_indices
        ],

        "test_end_dates": end_dates[
            test_indices
        ],

        "test_start_date": test_start_date,
    }

    return split


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_features(
    split: dict,
) -> tuple[dict, StandardScaler]:
    """
    Standardize input features.

    IMPORTANT:

    scaler is fitted ONLY
    on training data.
    """

    scaler = StandardScaler()

    X_train = split[
        "X_train"
    ]

    number_features = (
        X_train.shape[2]
    )

    flattened_train = (
        X_train.reshape(
            -1,
            number_features,
        )
    )

    scaler.fit(
        flattened_train
    )

    for key in [
        "X_train",
        "X_val",
        "X_test",
    ]:

        data = split[key]

        original_shape = (
            data.shape
        )

        flattened = (
            data.reshape(
                -1,
                number_features,
            )
        )

        transformed = (
            scaler.transform(
                flattened
            )
        )

        split[key] = (
            transformed
            .reshape(
                original_shape
            )
            .astype(
                np.float32
            )
        )

    return (
        split,
        scaler,
    )


# ============================================================
# TARGET TRANSFORMATION
# ============================================================

def transform_targets(
    split: dict,
) -> dict:
    """
    Apply log1p to realized variance.

    Raw squared returns are highly skewed.

    We train using:

        log(1 + variance)

    During evaluation:

        variance = exp(prediction) - 1
    """

    for key in [
        "y_train",
        "y_val",
        "y_test",
    ]:

        raw_key = (
            key + "_raw"
        )

        split[
            raw_key
        ] = (
            split[key]
            .copy()
        )

        split[key] = (
            np.log1p(
                split[key]
            )
            .astype(
                np.float32
            )
        )

    return split


# ============================================================
# DATE CONVERSION
# ============================================================

def dates_to_unicode_array(
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Convert pandas dates into safe Unicode arrays.

    This avoids NumPy object arrays
    and allows loading with:

        allow_pickle=False
    """

    return np.asarray(
        [
            date.strftime("%Y-%m-%d")
            for date in dates
        ],
        dtype="U10",
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    split: dict,
    scaler: StandardScaler,
    feature_names: list[str],
    target_names: list[str],
) -> None:
    """
    Save numerical arrays and metadata.
    """

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Safe date arrays
    # --------------------------------------------------------

    train_target_dates = (
        dates_to_unicode_array(
            split[
                "train_target_dates"
            ]
        )
    )

    val_target_dates = (
        dates_to_unicode_array(
            split[
                "val_target_dates"
            ]
        )
    )

    test_target_dates = (
        dates_to_unicode_array(
            split[
                "test_target_dates"
            ]
        )
    )

    train_end_dates = (
        dates_to_unicode_array(
            split[
                "train_end_dates"
            ]
        )
    )

    val_end_dates = (
        dates_to_unicode_array(
            split[
                "val_end_dates"
            ]
        )
    )

    test_end_dates = (
        dates_to_unicode_array(
            split[
                "test_end_dates"
            ]
        )
    )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_DATASET,

        X_train=split[
            "X_train"
        ],

        y_train=split[
            "y_train"
        ],

        y_train_raw=split[
            "y_train_raw"
        ],

        X_val=split[
            "X_val"
        ],

        y_val=split[
            "y_val"
        ],

        y_val_raw=split[
            "y_val_raw"
        ],

        X_test=split[
            "X_test"
        ],

        y_test=split[
            "y_test"
        ],

        y_test_raw=split[
            "y_test_raw"
        ],

        train_target_dates=(
            train_target_dates
        ),

        val_target_dates=(
            val_target_dates
        ),

        test_target_dates=(
            test_target_dates
        ),

        train_end_dates=(
            train_end_dates
        ),

        val_end_dates=(
            val_end_dates
        ),

        test_end_dates=(
            test_end_dates
        ),

        scaler_mean=(
            scaler.mean_
        ),

        scaler_scale=(
            scaler.scale_
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "sequence_length": (
            SEQUENCE_LENGTH
        ),

        "number_features": (
            len(feature_names)
        ),

        "number_targets": (
            len(target_names)
        ),

        "feature_names": (
            feature_names
        ),

        "target_names": (
            target_names
        ),

        "train_samples": int(
            len(
                split[
                    "X_train"
                ]
            )
        ),

        "validation_samples": int(
            len(
                split[
                    "X_val"
                ]
            )
        ),

        "test_samples": int(
            len(
                split[
                    "X_test"
                ]
            )
        ),

        "test_start_date": str(
            split[
                "test_start_date"
            ].date()
        ),

        "target_transformation": (
            "log1p(realized_variance)"
        ),

        "input_scale": (
            "StandardScaler fitted "
            "on training data only"
        ),

        "description": (
            "Multivariate financial "
            "time-series dataset for "
            "next-day variance forecasting"
        ),
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nFiles saved:")

    print(
        f"- {OUTPUT_DATASET}"
    )

    print(
        f"- {OUTPUT_METADATA}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    split: dict,
    feature_names: list[str],
    target_names: list[str],
) -> None:
    """
    Print dataset summary.
    """

    print("\n========================================")
    print("LSTM DATASET SUMMARY")
    print("========================================")

    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH} days"
    )

    print(
        f"Features        : "
        f"{len(feature_names)}"
    )

    print(
        f"Targets         : "
        f"{len(target_names)}"
    )

    print("\nArray shapes:")

    print(
        f"X_train : "
        f"{split['X_train'].shape}"
    )

    print(
        f"y_train : "
        f"{split['y_train'].shape}"
    )

    print(
        f"X_val   : "
        f"{split['X_val'].shape}"
    )

    print(
        f"y_val   : "
        f"{split['y_val'].shape}"
    )

    print(
        f"X_test  : "
        f"{split['X_test'].shape}"
    )

    print(
        f"y_test  : "
        f"{split['y_test'].shape}"
    )

    print("\nTarget periods:")

    print(
        f"Train : "
        f"{split['train_target_dates'][0].date()}"
        f" -> "
        f"{split['train_target_dates'][-1].date()}"
    )

    print(
        f"Val   : "
        f"{split['val_target_dates'][0].date()}"
        f" -> "
        f"{split['val_target_dates'][-1].date()}"
    )

    print(
        f"Test  : "
        f"{split['test_target_dates'][0].date()}"
        f" -> "
        f"{split['test_target_dates'][-1].date()}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    returns = load_returns()

    features = build_features(
        returns
    )

    (
        targets,
        target_dates,
    ) = build_targets(
        returns,
        features.index,
    )

    (
        X,
        y,
        end_dates,
        sequence_target_dates,
    ) = create_sequences(
        features,
        targets,
        target_dates,
    )

    split = temporal_split(
        X,
        y,
        end_dates,
        sequence_target_dates,
        returns,
    )

    split, scaler = (
        standardize_features(
            split
        )
    )

    split = transform_targets(
        split
    )

    feature_names = (
        features.columns.tolist()
    )

    target_names = (
        returns.columns.tolist()
    )

    save_dataset(
        split,
        scaler,
        feature_names,
        target_names,
    )

    print_summary(
        split,
        feature_names,
        target_names,
    )

    print(
        "\nLSTM dataset preparation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()