from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GARCH_FILE = (
    ROOT_DIR
    / "results"
    / "garch"
    / "garch_forecasts.csv"
)

LSTM_FILE = (
    ROOT_DIR
    / "results"
    / "lstm"
    / "lstm_forecasts.csv"
)

HYBRID_FILE = (
    ROOT_DIR
    / "results"
    / "hybrid"
    / "hybrid_forecasts.csv"
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

ASSET = "BNP"


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    # --------------------------------------------------------
    # GARCH
    # --------------------------------------------------------

    garch = pd.read_csv(
        GARCH_FILE
    )

    garch["Date"] = pd.to_datetime(
        garch["Date"]
    )

    garch = garch[
        garch["asset"] == ASSET
    ].copy()

    garch = garch[
        [
            "Date",
            "realized_variance",
            "predicted_variance",
        ]
    ]

    garch = garch.rename(
        columns={
            "Date": "date",
            "predicted_variance":
                "garch_variance",
        }
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm = pd.read_csv(
        LSTM_FILE
    )

    lstm["date"] = pd.to_datetime(
        lstm["date"]
    )

    lstm = lstm[
        lstm["asset"] == ASSET
    ].copy()

    lstm = lstm[
        [
            "date",
            "predicted_variance",
        ]
    ]

    lstm = lstm.rename(
        columns={
            "predicted_variance":
                "lstm_variance",
        }
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid = pd.read_csv(
        HYBRID_FILE
    )

    hybrid["date"] = pd.to_datetime(
        hybrid["date"]
    )

    hybrid = hybrid[
        hybrid["asset"] == ASSET
    ].copy()

    hybrid = hybrid[
        [
            "date",
            "hybrid_variance",
        ]
    ]

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    dataframe = (
        garch
        .merge(
            lstm,
            on="date",
            how="inner",
        )
        .merge(
            hybrid,
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# BASE PLOT
# ============================================================

def create_plot(
    dataframe: pd.DataFrame,
    zoom: bool,
) -> tuple:

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        dataframe["date"],
        dataframe["realized_variance"],
        label="Realized variance",
        linewidth=1.0,
        alpha=0.45,
    )

    ax.plot(
        dataframe["date"],
        dataframe["garch_variance"],
        label="GARCH",
        linewidth=1.4,
    )

    ax.plot(
        dataframe["date"],
        dataframe["lstm_variance"],
        label="LSTM",
        linewidth=1.4,
    )

    ax.plot(
        dataframe["date"],
        dataframe["hybrid_variance"],
        label="HYBRID",
        linewidth=1.6,
    )

    if zoom:

        ax.set_title(
            f"{ASSET} — Realized vs Forecast Variance "
            f"(Zoom 0–15 %²)"
        )

        ax.set_ylim(
            0,
            15,
        )

    else:

        ax.set_title(
            f"{ASSET} — Realized vs Forecast Variance"
        )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Daily variance (%²)"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    return fig, ax


# ============================================================
# SAVE GLOBAL FIGURE
# ============================================================

def save_global_plot(
    dataframe: pd.DataFrame,
) -> None:

    fig, _ = create_plot(
        dataframe,
        zoom=False,
    )

    output_file = (
        FIGURES_DIR
        / f"{ASSET.lower()}_variance_forecasts.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Global figure saved : "
        f"{output_file}"
    )

    plt.close(
        fig
    )


# ============================================================
# SAVE ZOOMED FIGURE
# ============================================================

def save_zoom_plot(
    dataframe: pd.DataFrame,
) -> None:

    fig, _ = create_plot(
        dataframe,
        zoom=True,
    )

    output_file = (
        FIGURES_DIR
        / f"{ASSET.lower()}_variance_forecasts_zoom.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Zoom figure saved   : "
        f"{output_file}"
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    dataframe = load_data()

    print(
        "\n========================================"
    )

    print(
        "VOLATILITY FORECAST FIGURES"
    )

    print(
        "========================================"
    )

    print(
        f"Asset        : {ASSET}"
    )

    print(
        f"Observations : {len(dataframe)}"
    )

    print(
        f"Start date   : "
        f"{dataframe['date'].min().date()}"
    )

    print(
        f"End date     : "
        f"{dataframe['date'].max().date()}"
    )

    print()

    save_global_plot(
        dataframe
    )

    save_zoom_plot(
        dataframe
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()