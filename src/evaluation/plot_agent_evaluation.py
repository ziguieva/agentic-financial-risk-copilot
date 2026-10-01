import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

SUMMARY_FILE = (
    ROOT_DIR
    / "results"
    / "agent_evaluation"
    / "agent_evaluation_summary_v3.json"
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
# LOAD RESULTS
# ============================================================

def load_summary() -> dict:

    with open(
        SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# PLOT
# ============================================================

def plot_guardrail_effect(
    summary: dict,
) -> None:

    categories = [
        "Single-call\ncompliance",
        "Argument\naccuracy",
        "Execution\naccuracy",
    ]

    raw_values = [
        summary[
            "raw_single_call_compliance_pct"
        ],
        summary[
            "top1_raw_argument_accuracy_pct"
        ],
        summary[
            "raw_strict_call_accuracy_pct"
        ],
    ]

    guarded_values = [
        summary[
            "guarded_single_call_compliance_pct"
        ],
        summary[
            "top1_guarded_argument_accuracy_pct"
        ],
        summary[
            "final_execution_accuracy_pct"
        ],
    ]

    positions = list(
        range(len(categories))
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    raw_bars = ax.bar(
        [
            x - width / 2
            for x in positions
        ],
        raw_values,
        width=width,
        label="Raw LLM",
    )

    guarded_bars = ax.bar(
        [
            x + width / 2
            for x in positions
        ],
        guarded_values,
        width=width,
        label="After guardrails",
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_title(
        "Agentic AI — Impact of Guardrails"
    )

    ax.set_xlabel(
        "Evaluation Metric"
    )

    ax.set_ylabel(
        "Accuracy / Compliance (%)"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        categories
    )

    ax.set_ylim(
        85,
        102,
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
        raw_bars,
        guarded_bars,
    ):

        for bar in bars:

            value = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.25,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # --------------------------------------------------------
    # ROUTING INFORMATION
    # --------------------------------------------------------

    routing = summary[
        "top1_tool_accuracy_pct"
    ]

    ax.text(
        0.5,
        0.03,
        (
            f"Top-1 tool routing accuracy: "
            f"{routing:.1f}% "
            f"({summary['total_cases']} test cases)"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        FIGURES_DIR
        / "agent_guardrails_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "\n========================================"
    )

    print(
        "AGENTIC AI — GUARDRAIL EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Top-1 tool accuracy           : "
        f"{routing:.1f}%"
    )

    print(
        f"Raw single-call compliance    : "
        f"{raw_values[0]:.1f}%"
    )

    print(
        f"After guardrails              : "
        f"{guarded_values[0]:.1f}%"
    )

    print()

    print(
        f"Raw argument accuracy         : "
        f"{raw_values[1]:.1f}%"
    )

    print(
        f"After guardrails              : "
        f"{guarded_values[1]:.1f}%"
    )

    print()

    print(
        f"Raw strict execution          : "
        f"{raw_values[2]:.1f}%"
    )

    print(
        f"Final execution accuracy      : "
        f"{guarded_values[2]:.1f}%"
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

    summary = load_summary()

    plot_guardrail_effect(
        summary
    )


if __name__ == "__main__":
    main()