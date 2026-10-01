from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

BACKTEST_FILE = (
    ROOT_DIR
    / "results"
    / "risk"
    / "var_backtest_summary.csv"
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
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    dataframe = pd.read_csv(
        BACKTEST_FILE
    )

    return dataframe


# ============================================================
# LABELS
# ============================================================

def build_labels(
    dataframe: pd.DataFrame,
) -> list[str]:

    labels = []

    for _, row in dataframe.iterrows():

        confidence = float(
            row["confidence"]
        )

        if confidence <= 1.0:
            confidence *= 100.0

        labels.append(
            f"{row['model']}\nVaR {confidence:.0f}%"
        )

    return labels


# ============================================================
# PLOT
# ============================================================

def plot_backtest(
    dataframe: pd.DataFrame,
) -> None:

    labels = build_labels(
        dataframe
    )

    positions = list(
        range(len(dataframe))
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # EXPECTED VIOLATIONS
    # --------------------------------------------------------

    expected_bars = ax.bar(
        [
            x - width / 2
            for x in positions
        ],
        dataframe[
            "expected_violations"
        ],
        width=width,
        label="Expected violations",
    )

    # --------------------------------------------------------
    # OBSERVED VIOLATIONS
    # --------------------------------------------------------

    observed_bars = ax.bar(
        [
            x + width / 2
            for x in positions
        ],
        dataframe[
            "violations"
        ],
        width=width,
        label="Observed violations",
    )

    # --------------------------------------------------------
    # TITLE / AXES
    # --------------------------------------------------------

    ax.set_title(
        "VaR Backtesting — Expected vs Observed Violations"
    )

    ax.set_xlabel(
        "Model and Confidence Level"
    )

    ax.set_ylabel(
        "Number of Violations"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # --------------------------------------------------------
    # VALUES ABOVE BARS
    # --------------------------------------------------------

    for bar in expected_bars:

        height = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.8,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in observed_bars:

        height = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.8,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # KUPIEC STATUS
    # --------------------------------------------------------

    for position, row in dataframe.iterrows():

        pvalue = float(
            row["kupiec_pvalue"]
        )

        accepted = bool(
            row["coverage_accepted_5pct"]
        )

        status = (
            "Not rejected"
            if accepted
            else "Rejected"
        )

        max_height = max(
            float(
                row["violations"]
            ),
            float(
                row["expected_violations"]
            ),
        )

        ax.text(
            position,
            max_height + 5,
            (
                f"Kupiec p={pvalue:.3f}\n"
                f"{status}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --------------------------------------------------------
    # Y LIMIT
    # --------------------------------------------------------

    max_observed = float(
        dataframe[
            "violations"
        ].max()
    )

    ax.set_ylim(
        0,
        max_observed * 1.25,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        FIGURES_DIR
        / "var_backtest_violations.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "VAR BACKTEST — EXPECTED VS OBSERVED"
    )

    print(
        "========================================"
    )

    print(
        dataframe[
            [
                "model",
                "confidence",
                "observations",
                "violations",
                "expected_violations",
                "violation_rate",
                "kupiec_pvalue",
                "coverage_accepted_5pct",
            ]
        ].to_string(
            index=False
        )
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

    plot_backtest(
        dataframe
    )


if __name__ == "__main__":
    main()