from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RISK_FILE = (
    ROOT_DIR
    / "results"
    / "risk"
    / "portfolio_risk_forecasts.csv"
)

FIGURES_DIR = (
    ROOT_DIR
    / "results"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "HYBRID"


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    dataframe = pd.read_csv(
        RISK_FILE
    )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"]
    )

    dataframe = dataframe[
        dataframe["model"] == MODEL
    ].copy()

    dataframe = (
        dataframe
        .sort_values("date")
        .reset_index(drop=True)
    )

    if dataframe.empty:

        raise ValueError(
            f"No observations found for model {MODEL}."
        )

    return dataframe


# ============================================================
# PLOT
# ============================================================

def plot_portfolio_risk(
    dataframe: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    # --------------------------------------------------------
    # DAILY VOLATILITY
    # --------------------------------------------------------

    ax.plot(
        dataframe["date"],
        dataframe[
            "predicted_volatility_daily_pct"
        ],
        linewidth=1.6,
        label="Predicted daily volatility",
    )

    # --------------------------------------------------------
    # VAR 95
    # --------------------------------------------------------

    ax.plot(
        dataframe["date"],
        dataframe["var_95_pct"],
        linewidth=1.6,
        label="VaR 95%",
    )

    # --------------------------------------------------------
    # VAR 99
    # --------------------------------------------------------

    ax.plot(
        dataframe["date"],
        dataframe["var_99_pct"],
        linewidth=1.8,
        label="VaR 99%",
    )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    ax.set_title(
        "Portfolio Risk Evolution — HYBRID Model"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Daily Risk (%)"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        FIGURES_DIR
        / "portfolio_risk_evolution_hybrid.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    latest = dataframe.iloc[-1]

    print(
        "\n========================================"
    )

    print(
        "PORTFOLIO RISK EVOLUTION"
    )

    print(
        "========================================"
    )

    print(
        f"Model        : {MODEL}"
    )

    print(
        f"Observations : {len(dataframe)}"
    )

    print(
        f"Period       : "
        f"{dataframe['date'].min().date()} "
        f"to "
        f"{dataframe['date'].max().date()}"
    )

    print()

    print(
        f"Latest date       : "
        f"{latest['date'].date()}"
    )

    print(
        f"Daily volatility  : "
        f"{latest['predicted_volatility_daily_pct']:.4f} %"
    )

    print(
        f"VaR 95 %          : "
        f"{latest['var_95_pct']:.4f} %"
    )

    print(
        f"VaR 99 %          : "
        f"{latest['var_99_pct']:.4f} %"
    )

    print()

    print(
        f"Saved to: {output_file}"
    )

    print(
        "========================================"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    dataframe = load_data()

    plot_portfolio_risk(
        dataframe
    )


if __name__ == "__main__":
    main()