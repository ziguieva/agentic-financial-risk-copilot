from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GARCH_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "garch"
    / "garch_metrics.csv"
)

LSTM_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "lstm"
    / "lstm_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "comparison"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "garch_vs_lstm.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GARCH and LSTM evaluation metrics.
    """

    if not GARCH_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"GARCH metrics not found: {GARCH_METRICS_FILE}"
        )

    if not LSTM_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"LSTM metrics not found: {LSTM_METRICS_FILE}"
        )

    garch = pd.read_csv(
        GARCH_METRICS_FILE
    )

    lstm = pd.read_csv(
        LSTM_METRICS_FILE
    )

    print("Metrics loaded successfully.")

    return garch, lstm


# ============================================================
# BUILD COMPARISON
# ============================================================

def build_comparison(
    garch: pd.DataFrame,
    lstm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge GARCH and LSTM metrics.

    Lower is better:
        MSE
        MAE
        QLIKE

    Higher is better:
        correlation
    """

    garch_selected = garch[
        [
            "asset",
            "mse",
            "mae",
            "qlike",
            "correlation",
        ]
    ].copy()

    lstm_selected = lstm[
        [
            "asset",
            "mse",
            "mae",
            "qlike",
            "correlation",
        ]
    ].copy()

    garch_selected = (
        garch_selected.rename(
            columns={
                "mse": "garch_mse",
                "mae": "garch_mae",
                "qlike": "garch_qlike",
                "correlation": "garch_correlation",
            }
        )
    )

    lstm_selected = (
        lstm_selected.rename(
            columns={
                "mse": "lstm_mse",
                "mae": "lstm_mae",
                "qlike": "lstm_qlike",
                "correlation": "lstm_correlation",
            }
        )
    )

    comparison = pd.merge(
        garch_selected,
        lstm_selected,
        on="asset",
        how="inner",
    )

    # ========================================================
    # Relative improvements
    # ========================================================

    comparison[
        "mae_improvement_pct"
    ] = (
        (
            comparison["garch_mae"]
            - comparison["lstm_mae"]
        )
        / comparison["garch_mae"]
        * 100.0
    )

    comparison[
        "mse_improvement_pct"
    ] = (
        (
            comparison["garch_mse"]
            - comparison["lstm_mse"]
        )
        / comparison["garch_mse"]
        * 100.0
    )

    comparison[
        "qlike_improvement_pct"
    ] = (
        (
            comparison["garch_qlike"]
            - comparison["lstm_qlike"]
        )
        / comparison["garch_qlike"]
        * 100.0
    )

    comparison[
        "correlation_improvement"
    ] = (
        comparison["lstm_correlation"]
        - comparison["garch_correlation"]
    )

    # ========================================================
    # Best model by metric
    # ========================================================

    comparison[
        "best_mae"
    ] = comparison.apply(
        lambda row:
        "LSTM"
        if row["lstm_mae"] < row["garch_mae"]
        else "GARCH",
        axis=1,
    )

    comparison[
        "best_mse"
    ] = comparison.apply(
        lambda row:
        "LSTM"
        if row["lstm_mse"] < row["garch_mse"]
        else "GARCH",
        axis=1,
    )

    comparison[
        "best_qlike"
    ] = comparison.apply(
        lambda row:
        "LSTM"
        if row["lstm_qlike"] < row["garch_qlike"]
        else "GARCH",
        axis=1,
    )

    comparison[
        "best_correlation"
    ] = comparison.apply(
        lambda row:
        "LSTM"
        if row["lstm_correlation"] > row["garch_correlation"]
        else "GARCH",
        axis=1,
    )

    return comparison


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """
    Print model comparison.
    """

    print("\n========================================")
    print("GARCH VS LSTM")
    print("========================================")

    display_columns = [
        "asset",
        "garch_mae",
        "lstm_mae",
        "mae_improvement_pct",
        "garch_qlike",
        "lstm_qlike",
        "garch_correlation",
        "lstm_correlation",
    ]

    print(
        comparison[
            display_columns
        ].round(6)
    )

    print("\n========================================")
    print("WIN COUNTS")
    print("========================================")

    metrics = {
        "MAE": "best_mae",
        "MSE": "best_mse",
        "QLIKE": "best_qlike",
        "Correlation": "best_correlation",
    }

    for metric_name, column in metrics.items():

        counts = (
            comparison[column]
            .value_counts()
        )

        garch_wins = int(
            counts.get(
                "GARCH",
                0,
            )
        )

        lstm_wins = int(
            counts.get(
                "LSTM",
                0,
            )
        )

        print(
            f"{metric_name:<12} | "
            f"GARCH: {garch_wins} | "
            f"LSTM: {lstm_wins}"
        )


# ============================================================
# GLOBAL RESULTS
# ============================================================

def print_global_results(
    comparison: pd.DataFrame,
) -> None:
    """
    Print average improvements across assets.
    """

    print("\n========================================")
    print("AVERAGE IMPROVEMENTS")
    print("========================================")

    mean_mae_improvement = (
        comparison[
            "mae_improvement_pct"
        ].mean()
    )

    mean_mse_improvement = (
        comparison[
            "mse_improvement_pct"
        ].mean()
    )

    mean_qlike_improvement = (
        comparison[
            "qlike_improvement_pct"
        ].mean()
    )

    mean_correlation_change = (
        comparison[
            "correlation_improvement"
        ].mean()
    )

    print(
        f"MAE improvement        : "
        f"{mean_mae_improvement:.2f}%"
    )

    print(
        f"MSE improvement        : "
        f"{mean_mse_improvement:.2f}%"
    )

    print(
        f"QLIKE improvement      : "
        f"{mean_qlike_improvement:.2f}%"
    )

    print(
        f"Correlation change     : "
        f"{mean_correlation_change:+.4f}"
    )


# ============================================================
# SAVE
# ============================================================

def save_results(
    comparison: pd.DataFrame,
) -> None:
    """
    Save comparison table.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nComparison saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    garch, lstm = load_metrics()

    comparison = build_comparison(
        garch,
        lstm,
    )

    print_summary(
        comparison
    )

    print_global_results(
        comparison
    )

    save_results(
        comparison
    )

    print(
        "\nModel comparison completed successfully."
    )


if __name__ == "__main__":
    main()