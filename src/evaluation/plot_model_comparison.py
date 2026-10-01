from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GARCH_FILE = ROOT_DIR / "results" / "garch" / "garch_metrics.csv"
LSTM_FILE = ROOT_DIR / "results" / "lstm" / "lstm_metrics.csv"
HYBRID_FILE = ROOT_DIR / "results" / "hybrid" / "hybrid_metrics.csv"

FIGURES_DIR = ROOT_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_metrics(metric: str) -> pd.DataFrame:

    garch = pd.read_csv(GARCH_FILE)[
        ["asset", metric]
    ].rename(columns={metric: "GARCH"})

    lstm = pd.read_csv(LSTM_FILE)[
        ["asset", metric]
    ].rename(columns={metric: "LSTM"})

    hybrid = pd.read_csv(HYBRID_FILE)[
        ["asset", metric]
    ].rename(columns={metric: "HYBRID"})

    return (
        garch
        .merge(lstm, on="asset")
        .merge(hybrid, on="asset")
    )


# ============================================================
# GENERIC BAR CHART
# ============================================================

def plot_metric(
    dataframe: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    output_name: str,
) -> None:

    positions = list(
        range(len(dataframe))
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        [x - width for x in positions],
        dataframe["GARCH"],
        width=width,
        label="GARCH",
    )

    ax.bar(
        positions,
        dataframe["LSTM"],
        width=width,
        label="LSTM",
    )

    ax.bar(
        [x + width for x in positions],
        dataframe["HYBRID"],
        width=width,
        label="HYBRID",
    )

    ax.set_title(
        f"Volatility Forecasting — {metric_name} Comparison"
    )

    ax.set_xlabel(
        "Asset"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        dataframe["asset"]
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / output_name
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"{metric_name} figure saved: "
        f"{output_file}"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n========================================"
    )

    print(
        "MODEL COMPARISON FIGURES"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse_data = load_metrics(
        "mse"
    )

    plot_metric(
        dataframe=mse_data,
        metric_name="MSE",
        ylabel="Mean Squared Error",
        output_name="model_comparison_mse.png",
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae_data = load_metrics(
        "mae"
    )

    plot_metric(
        dataframe=mae_data,
        metric_name="MAE",
        ylabel="Mean Absolute Error",
        output_name="model_comparison_mae.png",
    )

    print(
        "========================================"
    )

    print(
        "\nMAE values:"
    )

    print(
        mae_data.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # QLIKE
    # --------------------------------------------------------

    qlike_data = load_metrics(
        "qlike"
    )

    plot_metric(
        dataframe=qlike_data,
        metric_name="QLIKE",
        ylabel="QLIKE Loss",
        output_name="model_comparison_qlike.png",
    )

    print(
        "\nQLIKE values:"
    )

    print(
        qlike_data.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    correlation_data = load_metrics(
        "correlation"
    )

    plot_metric(
        dataframe=correlation_data,
        metric_name="Correlation",
        ylabel="Correlation with Realized Variance",
        output_name="model_comparison_correlation.png",
    )

    print(
        "\nCorrelation values:"
    )

    print(
        correlation_data.to_string(
            index=False
        )
    )
if __name__ == "__main__":
    main()