from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

HISTORY_FILE = (
    ROOT_DIR
    / "results"
    / "lstm"
    / "training_history.csv"
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

def load_history() -> pd.DataFrame:

    return pd.read_csv(
        HISTORY_FILE
    )


# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(
    dataframe: pd.DataFrame,
) -> None:

    # --------------------------------------------------------
    # BEST VALIDATION EPOCH
    # --------------------------------------------------------

    best_index = (
        dataframe["val_loss"]
        .idxmin()
    )

    best_epoch = int(
        dataframe.loc[
            best_index,
            "epoch"
        ]
    )

    best_val_loss = float(
        dataframe.loc[
            best_index,
            "val_loss"
        ]
    )

    final_epoch = int(
        dataframe["epoch"].max()
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.plot(
        dataframe["epoch"],
        dataframe["train_loss"],
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="Training loss",
    )

    ax.plot(
        dataframe["epoch"],
        dataframe["val_loss"],
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="Validation loss",
    )

    # --------------------------------------------------------
    # BEST EPOCH
    # --------------------------------------------------------

    ax.axvline(
        best_epoch,
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch = {best_epoch}",
    )

    ax.scatter(
        best_epoch,
        best_val_loss,
        s=70,
        zorder=5,
    )

    ax.annotate(
        (
            f"Best validation loss\n"
            f"Epoch {best_epoch}: "
            f"{best_val_loss:.6f}"
        ),
        xy=(
            best_epoch,
            best_val_loss,
        ),
        xytext=(
            best_epoch + 2,
            best_val_loss + 0.06,
        ),
        arrowprops={
            "arrowstyle": "->",
        },
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    ax.axvline(
        final_epoch,
        linestyle=":",
        alpha=0.7,
        label=f"Training stopped = epoch {final_epoch}",
    )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    ax.set_title(
        "LSTM Training — Training and Validation Loss"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss (MSE on log1p variance)"
    )

    ax.set_xticks(
        dataframe["epoch"]
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
        / "lstm_training_history.png"
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
        "LSTM TRAINING HISTORY"
    )

    print(
        "========================================"
    )

    print(
        f"Best epoch          : {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Final epoch         : {final_epoch}"
    )

    print(
        f"Saved to            : {output_file}"
    )

    print(
        "========================================"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    dataframe = load_history()

    plot_training_history(
        dataframe
    )


if __name__ == "__main__":
    main()