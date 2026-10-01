from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "results" / "hybrid" / "hybrid_weights.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "hybrid_model_weights.png"


# ============================================================
# LOAD DATA
# ============================================================

def load_weights() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "asset",
        "lstm_weight",
        "garch_weight",
        "validation_garch_qlike",
        "validation_lstm_qlike",
        "validation_hybrid_qlike",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in hybrid_weights.csv: {missing}"
        )

    return df


# ============================================================
# PLOT
# ============================================================

def plot_hybrid_weights(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = df["asset"].tolist()
    garch_weights = df["garch_weight"].tolist()
    lstm_weights = df["lstm_weight"].tolist()

    positions = range(len(assets))

    plt.figure(figsize=(12, 7))

    bars_garch = plt.bar(
        positions,
        garch_weights,
        label="GARCH weight",
    )

    bars_lstm = plt.bar(
        positions,
        lstm_weights,
        bottom=garch_weights,
        label="LSTM weight",
    )

    plt.xticks(positions, assets, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel("Weight in hybrid model", fontsize=12)
    plt.xlabel("Asset", fontsize=12)
    plt.title(
        "Hybrid Model — GARCH vs LSTM Weights by Asset",
        fontsize=16,
    )
    plt.ylim(0, 1.15)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=11)

    # --------------------------------------------------------
    # ANNOTATIONS INSIDE BARS
    # --------------------------------------------------------

    for i, (g, l) in enumerate(zip(garch_weights, lstm_weights)):
        if g > 0.08:
            plt.text(
                i,
                g / 2,
                f"{g:.0%}",
                ha="center",
                va="center",
                fontsize=10,
            )

        if l > 0.08:
            plt.text(
                i,
                g + l / 2,
                f"{l:.0%}",
                ha="center",
                va="center",
                fontsize=10,
            )

        dominant_model = "GARCH" if g > l else "LSTM" if l > g else "Balanced"

        plt.text(
            i,
            1.04,
            dominant_model,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # FOOTNOTE WITH BEST VALIDATION QLIKE
    # --------------------------------------------------------

    note_lines = []
    for _, row in df.iterrows():
        note_lines.append(
            f"{row['asset']}: hybrid QLIKE={row['validation_hybrid_qlike']:.3f}"
        )

    footnote = " | ".join(note_lines)

    plt.figtext(
        0.5,
        0.01,
        footnote,
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {OUTPUT_PATH}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_weights()

    print("Columns:")
    print(df.columns.tolist())
    print("\nData:")
    print(df.to_string(index=False))

    plot_hybrid_weights(df)


if __name__ == "__main__":
    main()
