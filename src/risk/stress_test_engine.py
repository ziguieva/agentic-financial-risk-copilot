from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PORTFOLIO
# ============================================================

PORTFOLIO_WEIGHTS = {
    "BNP": 1.0 / 3.0,
    "AXA": 1.0 / 3.0,
    "SOCIETE_GENERALE": 1.0 / 3.0,
}


# ============================================================
# HYPOTHETICAL STRESS SCENARIOS
#
# IMPORTANT:
# These are illustrative scenarios.
# They are NOT forecasts.
#
# shock values are percentage price changes.
# Example:
# -10 means a 10% fall.
# ============================================================

STRESS_SCENARIOS = {

    "MARKET_CORRECTION": {

        "description":
            "Broad but moderate equity market correction.",

        "shocks": {
            "BNP": -6.0,
            "AXA": -5.0,
            "SOCIETE_GENERALE": -8.0,
        },

        "volatility_multiplier": 1.50,
    },


    "RATE_SHOCK": {

        "description":
            "Sharp interest-rate repricing affecting financial stocks.",

        "shocks": {
            "BNP": -7.0,
            "AXA": -9.0,
            "SOCIETE_GENERALE": -8.0,
        },

        "volatility_multiplier": 1.70,
    },


    "BANKING_STRESS": {

        "description":
            "Severe banking-sector stress scenario.",

        "shocks": {
            "BNP": -12.0,
            "AXA": -6.0,
            "SOCIETE_GENERALE": -15.0,
        },

        "volatility_multiplier": 2.00,
    },


    "SEVERE_CRASH": {

        "description":
            "Extreme simultaneous financial-market shock.",

        "shocks": {
            "BNP": -18.0,
            "AXA": -15.0,
            "SOCIETE_GENERALE": -22.0,
        },

        "volatility_multiplier": 2.50,
    },
}


# ============================================================
# INPUT FILE
# ============================================================

RISK_FORECAST_FILE = (
    PROJECT_ROOT
    / "results"
    / "risk"
    / "portfolio_risk_forecasts.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stress"
)

STRESS_RESULTS_FILE = (
    RESULTS_DIR
    / "stress_test_results.csv"
)

CONTRIBUTIONS_FILE = (
    RESULTS_DIR
    / "stress_test_contributions.csv"
)

CONFIG_FILE = (
    RESULTS_DIR
    / "stress_test_config.json"
)


# ============================================================
# LOAD LATEST RISK FORECAST
# ============================================================

def load_latest_hybrid_risk() -> pd.Series:
    """
    Load latest Hybrid portfolio risk forecast.
    """

    if not RISK_FORECAST_FILE.exists():

        raise FileNotFoundError(
            f"Risk forecast file not found: "
            f"{RISK_FORECAST_FILE}"
        )

    data = pd.read_csv(
        RISK_FORECAST_FILE,
        parse_dates=["date"],
    )

    hybrid = (
        data[
            data["model"] == "HYBRID"
        ]
        .sort_values("date")
    )

    if hybrid.empty:

        raise RuntimeError(
            "No HYBRID risk forecasts found."
        )

    latest = hybrid.iloc[-1]

    return latest


# ============================================================
# VALIDATE PORTFOLIO
# ============================================================

def validate_portfolio() -> tuple[list[str], np.ndarray]:
    """
    Validate portfolio weights.
    """

    assets = list(
        PORTFOLIO_WEIGHTS.keys()
    )

    weights = np.array(
        [
            PORTFOLIO_WEIGHTS[asset]
            for asset in assets
        ],
        dtype=float,
    )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):

        raise ValueError(
            "Portfolio weights must sum to 1."
        )

    return assets, weights


# ============================================================
# SCENARIO PORTFOLIO LOSS
# ============================================================

def calculate_scenario_loss(
    scenario: dict,
    assets: list[str],
    weights: np.ndarray,
) -> tuple[float, dict]:
    """
    Calculate deterministic portfolio loss
    under a stress scenario.

    Portfolio return:

        sum(weight_i * shock_i)

    Stress loss is returned as a POSITIVE
    percentage loss.
    """

    contributions = {}

    portfolio_return = 0.0

    for asset, weight in zip(
        assets,
        weights,
    ):

        shock = scenario[
            "shocks"
        ][asset]

        weighted_return = (
            weight * shock
        )

        portfolio_return += (
            weighted_return
        )

        contributions[asset] = (
            -weighted_return
        )

    stress_loss = (
        -portfolio_return
    )

    return (
        stress_loss,
        contributions,
    )


# ============================================================
# STRESSED RISK METRICS
# ============================================================

def calculate_stressed_metrics(
    latest_risk: pd.Series,
    volatility_multiplier: float,
) -> dict:
    """
    Scale volatility-based risk measures.

    VaR and Expected Shortfall scale linearly
    with volatility under this simplified
    stress framework.
    """

    daily_vol = (
        latest_risk[
            "predicted_volatility_daily_pct"
        ]
    )

    annual_vol = (
        latest_risk[
            "predicted_volatility_annualized"
        ]
    )

    stressed_daily_vol = (
        daily_vol
        * volatility_multiplier
    )

    stressed_annual_vol = (
        annual_vol
        * volatility_multiplier
    )

    stressed_var_95 = (
        latest_risk[
            "var_95_pct"
        ]
        * volatility_multiplier
    )

    stressed_es_95 = (
        latest_risk[
            "es_95_pct"
        ]
        * volatility_multiplier
    )

    stressed_var_99 = (
        latest_risk[
            "var_99_pct"
        ]
        * volatility_multiplier
    )

    stressed_es_99 = (
        latest_risk[
            "es_99_pct"
        ]
        * volatility_multiplier
    )

    return {

        "stressed_daily_volatility_pct":
            stressed_daily_vol,

        "stressed_annualized_volatility":
            stressed_annual_vol,

        "stressed_var_95_pct":
            stressed_var_95,

        "stressed_es_95_pct":
            stressed_es_95,

        "stressed_var_99_pct":
            stressed_var_99,

        "stressed_es_99_pct":
            stressed_es_99,
    }


# ============================================================
# RUN STRESS TESTS
# ============================================================

def run_stress_tests(
    latest_risk: pd.Series,
    assets: list[str],
    weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate every stress scenario.
    """

    results = []
    contribution_rows = []

    print("\n========================================")
    print("STRESS TESTING")
    print("========================================")

    for scenario_name, scenario in (
        STRESS_SCENARIOS.items()
    ):

        (
            stress_loss,
            contributions,
        ) = calculate_scenario_loss(

            scenario,
            assets,
            weights,
        )

        stressed_metrics = (
            calculate_stressed_metrics(

                latest_risk,

                scenario[
                    "volatility_multiplier"
                ],
            )
        )

        # ----------------------------------------------------
        # Compare deterministic shock to current risk measures
        # ----------------------------------------------------

        var95_multiple = (
            stress_loss
            / latest_risk[
                "var_95_pct"
            ]
        )

        var99_multiple = (
            stress_loss
            / latest_risk[
                "var_99_pct"
            ]
        )

        es99_multiple = (
            stress_loss
            / latest_risk[
                "es_99_pct"
            ]
        )

        results.append(
            {

                "scenario": scenario_name,

                "description":
                    scenario[
                        "description"
                    ],

                "volatility_multiplier":
                    scenario[
                        "volatility_multiplier"
                    ],

                "portfolio_stress_loss_pct":
                    stress_loss,

                "current_var_95_pct":
                    latest_risk[
                        "var_95_pct"
                    ],

                "current_var_99_pct":
                    latest_risk[
                        "var_99_pct"
                    ],

                "current_es_99_pct":
                    latest_risk[
                        "es_99_pct"
                    ],

                "stress_loss_vs_var95":
                    var95_multiple,

                "stress_loss_vs_var99":
                    var99_multiple,

                "stress_loss_vs_es99":
                    es99_multiple,

                **stressed_metrics,
            }
        )

        # ----------------------------------------------------
        # Asset contributions
        # ----------------------------------------------------

        for asset in assets:

            contribution_rows.append(
                {

                    "scenario":
                        scenario_name,

                    "asset":
                        asset,

                    "portfolio_weight":
                        PORTFOLIO_WEIGHTS[
                            asset
                        ],

                    "asset_shock_pct":
                        scenario[
                            "shocks"
                        ][asset],

                    "loss_contribution_pct":
                        contributions[
                            asset
                        ],
                }
            )

        print(
            f"{scenario_name:<20} | "
            f"Portfolio loss: "
            f"{stress_loss:.2f}%"
        )

    return (
        pd.DataFrame(
            results
        ),
        pd.DataFrame(
            contribution_rows
        ),
    )


# ============================================================
# SAVE CONFIG
# ============================================================

def save_config() -> None:
    """
    Save stress-test configuration.
    """

    configuration = {

        "portfolio_weights":
            PORTFOLIO_WEIGHTS,

        "scenarios":
            STRESS_SCENARIOS,

        "methodology": (
            "Deterministic asset shocks "
            "combined with volatility scaling."
        ),

        "warning": (
            "Stress scenarios are hypothetical "
            "and are not forecasts."
        ),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    """
    Save stress-test outputs.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        STRESS_RESULTS_FILE,
        index=False,
    )

    contributions.to_csv(
        CONTRIBUTIONS_FILE,
        index=False,
    )

    save_config()

    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print(
        STRESS_RESULTS_FILE
    )

    print(
        CONTRIBUTIONS_FILE
    )

    print(
        CONFIG_FILE
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    latest_risk: pd.Series,
    results: pd.DataFrame,
) -> None:
    """
    Display key stress-testing results.
    """

    print("\n========================================")
    print("CURRENT HYBRID RISK")
    print("========================================")

    print(
        f"Date                  : "
        f"{latest_risk['date'].date()}"
    )

    print(
        f"Daily volatility      : "
        f"{latest_risk['predicted_volatility_daily_pct']:.3f}%"
    )

    print(
        f"VaR 95%               : "
        f"{latest_risk['var_95_pct']:.3f}%"
    )

    print(
        f"ES 95%                : "
        f"{latest_risk['es_95_pct']:.3f}%"
    )

    print(
        f"VaR 99%               : "
        f"{latest_risk['var_99_pct']:.3f}%"
    )

    print(
        f"ES 99%                : "
        f"{latest_risk['es_99_pct']:.3f}%"
    )

    print("\n========================================")
    print("STRESS TEST SUMMARY")
    print("========================================")

    columns = [

        "scenario",

        "portfolio_stress_loss_pct",

        "volatility_multiplier",

        "stress_loss_vs_var95",

        "stress_loss_vs_var99",

        "stress_loss_vs_es99",
    ]

    print(

        results[
            columns
        ]

        .round(3)

        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    latest_risk = (
        load_latest_hybrid_risk()
    )

    assets, weights = (
        validate_portfolio()
    )

    (
        results,
        contributions,
    ) = run_stress_tests(

        latest_risk,

        assets,

        weights,
    )

    save_results(
        results,
        contributions,
    )

    print_summary(
        latest_risk,
        results,
    )

    print(
        "\nStress testing completed successfully."
    )


if __name__ == "__main__":
    main()