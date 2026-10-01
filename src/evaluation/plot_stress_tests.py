from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

STRESS_FILE = (
    ROOT_DIR
    / "results"
    / "stress"
    / "stress_test_results.csv"
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

    return pd.read_csv(
        STRESS_FILE
    )


# ============================================================
# LABEL FORMAT
# ============================================================

def format_scenario_name(
    scenario: str,
) -> str:

    return scenario.replace(
        "_",
        "\n",
    )


# ============================================================
# FIGURE 1 — PORTFOLIO LOSS
# ============================================================

def plot_stress_losses(
    dataframe: pd.DataFrame,
) -> None:

    labels = [
        format_scenario_name(
            scenario
        )
        for scenario
        in dataframe["scenario"]
    ]

    values = dataframe[
        "portfolio_stress_loss_pct"
    ]

    positions = list(
        range(len(dataframe))
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        positions,
        values,
    )

    ax.set_title(
        "Portfolio Stress Testing — Hypothetical Loss by Scenario"
    )

    ax.set_xlabel(
        "Stress Scenario"
    )

    ax.set_ylabel(
        "Portfolio Loss (%)"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # Values above bars
    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(
        0,
        float(values.max()) * 1.20,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "stress_test_portfolio_losses.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Portfolio-loss figure saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# FIGURE 2 — STRESS LOSS VS RISK MEASURES
# ============================================================

def plot_stress_vs_risk(
    dataframe: pd.DataFrame,
) -> None:

    labels = [
        format_scenario_name(
            scenario
        )
        for scenario
        in dataframe["scenario"]
    ]

    positions = list(
        range(len(dataframe))
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars_var95 = ax.bar(
        [
            x - width
            for x in positions
        ],
        dataframe[
            "stress_loss_vs_var95"
        ],
        width=width,
        label="Stress loss / VaR 95%",
    )

    bars_var99 = ax.bar(
        positions,
        dataframe[
            "stress_loss_vs_var99"
        ],
        width=width,
        label="Stress loss / VaR 99%",
    )

    bars_es99 = ax.bar(
        [
            x + width
            for x in positions
        ],
        dataframe[
            "stress_loss_vs_es99"
        ],
        width=width,
        label="Stress loss / ES 99%",
    )

    ax.set_title(
        "Stress Loss Relative to VaR and Expected Shortfall"
    )

    ax.set_xlabel(
        "Stress Scenario"
    )

    ax.set_ylabel(
        "Multiple of Risk Measure"
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

    for bars in (
        bars_var95,
        bars_var99,
        bars_es99,
    ):

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.12,
                f"{height:.2f}×",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    max_value = max(
        dataframe[
            "stress_loss_vs_var95"
        ].max(),
        dataframe[
            "stress_loss_vs_var99"
        ].max(),
        dataframe[
            "stress_loss_vs_es99"
        ].max(),
    )

    ax.set_ylim(
        0,
        float(max_value) * 1.18,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "stress_test_vs_risk_measures.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Stress-vs-risk figure saved: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    dataframe = load_data()

    print(
        "\n========================================"
    )

    print(
        "STRESS TEST FIGURES"
    )

    print(
        "========================================"
    )

    print(
        dataframe[
            [
                "scenario",
                "portfolio_stress_loss_pct",
                "stress_loss_vs_var95",
                "stress_loss_vs_var99",
                "stress_loss_vs_es99",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    plot_stress_losses(
        dataframe
    )

    plot_stress_vs_risk(
        dataframe
    )


if __name__ == "__main__":
    main()