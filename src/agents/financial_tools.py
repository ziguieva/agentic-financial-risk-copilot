from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


RISK_FORECAST_FILE = (
    PROJECT_ROOT
    / "results"
    / "risk"
    / "portfolio_risk_forecasts.csv"
)

VAR_BACKTEST_FILE = (
    PROJECT_ROOT
    / "results"
    / "risk"
    / "var_backtest_summary.csv"
)

MODEL_COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "comparison"
    / "garch_vs_lstm.csv"
)

HYBRID_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "hybrid"
    / "hybrid_metrics.csv"
)

STRESS_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "stress"
    / "stress_test_results.csv"
)

STRESS_CONTRIBUTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "stress"
    / "stress_test_contributions.csv"
)


# ============================================================
# HELPER
# ============================================================

def check_file(
    file_path: Path,
) -> None:
    """
    Verify that a required result file exists.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path}"
        )


# ============================================================
# TOOL 1
# CURRENT PORTFOLIO RISK
# ============================================================

def get_current_risk() -> dict[str, Any]:
    """
    Return the latest HYBRID portfolio risk forecast.

    This tool provides:

    - daily volatility
    - annualized volatility
    - VaR 95%
    - Expected Shortfall 95%
    - VaR 99%
    - Expected Shortfall 99%

    Returns
    -------
    dict
        Latest portfolio risk metrics.
    """

    check_file(
        RISK_FORECAST_FILE
    )

    data = pd.read_csv(
        RISK_FORECAST_FILE,
        parse_dates=["date"],
    )

    hybrid = (
        data[
            data["model"]
            == "HYBRID"
        ]
        .sort_values(
            "date"
        )
    )

    if hybrid.empty:

        raise RuntimeError(
            "No HYBRID risk forecast found."
        )

    latest = (
        hybrid.iloc[-1]
    )

    return {

        "date": (
            latest["date"]
            .strftime("%Y-%m-%d")
        ),

        "model": "HYBRID",

        "daily_volatility_pct": round(
            float(
                latest[
                    "predicted_volatility_daily_pct"
                ]
            ),
            4,
        ),

        "annualized_volatility_pct": round(
            float(
                latest[
                    "predicted_volatility_annualized"
                ]
            )
            * 100.0,
            4,
        ),

        "var_95_pct": round(
            float(
                latest[
                    "var_95_pct"
                ]
            ),
            4,
        ),

        "expected_shortfall_95_pct": round(
            float(
                latest[
                    "es_95_pct"
                ]
            ),
            4,
        ),

        "var_99_pct": round(
            float(
                latest[
                    "var_99_pct"
                ]
            ),
            4,
        ),

        "expected_shortfall_99_pct": round(
            float(
                latest[
                    "es_99_pct"
                ]
            ),
            4,
        ),
    }


# ============================================================
# TOOL 2
# MODEL COMPARISON
# ============================================================

def compare_volatility_models() -> dict[str, Any]:
    """
    Compare GARCH, LSTM and HYBRID volatility models.

    Returns test metrics for each asset.
    """

    check_file(
        MODEL_COMPARISON_FILE
    )

    check_file(
        HYBRID_METRICS_FILE
    )

    comparison = pd.read_csv(
        MODEL_COMPARISON_FILE
    )

    hybrid = pd.read_csv(
        HYBRID_METRICS_FILE
    )

    results = []

    for _, row in comparison.iterrows():

        asset = row["asset"]

        hybrid_row = (
            hybrid[
                hybrid["asset"]
                == asset
            ]
        )

        if hybrid_row.empty:

            raise RuntimeError(
                f"Hybrid metrics missing "
                f"for {asset}."
            )

        hybrid_row = (
            hybrid_row.iloc[0]
        )

        results.append(
            {

                "asset": asset,

                "GARCH": {

                    "mse": round(
                        float(
                            row[
                                "garch_mse"
                            ]
                        ),
                        6,
                    ),

                    "mae": round(
                        float(
                            row[
                                "garch_mae"
                            ]
                        ),
                        6,
                    ),

                    "qlike": round(
                        float(
                            row[
                                "garch_qlike"
                            ]
                        ),
                        6,
                    ),

                    "correlation": round(
                        float(
                            row[
                                "garch_correlation"
                            ]
                        ),
                        6,
                    ),
                },

                "LSTM": {

                    "mse": round(
                        float(
                            row[
                                "lstm_mse"
                            ]
                        ),
                        6,
                    ),

                    "mae": round(
                        float(
                            row[
                                "lstm_mae"
                            ]
                        ),
                        6,
                    ),

                    "qlike": round(
                        float(
                            row[
                                "lstm_qlike"
                            ]
                        ),
                        6,
                    ),

                    "correlation": round(
                        float(
                            row[
                                "lstm_correlation"
                            ]
                        ),
                        6,
                    ),
                },

                "HYBRID": {

                    "mse": round(
                        float(
                            hybrid_row[
                                "mse"
                            ]
                        ),
                        6,
                    ),

                    "mae": round(
                        float(
                            hybrid_row[
                                "mae"
                            ]
                        ),
                        6,
                    ),

                    "qlike": round(
                        float(
                            hybrid_row[
                                "qlike"
                            ]
                        ),
                        6,
                    ),

                    "correlation": round(
                        float(
                            hybrid_row[
                                "correlation"
                            ]
                        ),
                        6,
                    ),
                },
            }
        )

    return {

        "test_period": (
            "2024-05-23 to 2026-09-28"
        ),

        "assets": results,

        "interpretation_note": (
            "Lower MSE, MAE and QLIKE are better. "
            "Higher correlation is better. "
            "No single metric should be used alone."
        ),
    }


# ============================================================
# TOOL 3
# VAR BACKTEST
# ============================================================

def get_var_backtest() -> dict[str, Any]:
    """
    Return VaR backtesting results.

    Includes:

    - violation counts
    - expected violations
    - violation rate
    - Kupiec test p-value
    """

    check_file(
        VAR_BACKTEST_FILE
    )

    data = pd.read_csv(
        VAR_BACKTEST_FILE
    )

    results = []

    for _, row in data.iterrows():

        results.append(
            {

                "model": row[
                    "model"
                ],

                "confidence_level": round(
                    float(
                        row[
                            "confidence"
                        ]
                    ),
                    2,
                ),

                "observations": int(
                    row[
                        "observations"
                    ]
                ),

                "violations": int(
                    row[
                        "violations"
                    ]
                ),

                "expected_violations": round(
                    float(
                        row[
                            "expected_violations"
                        ]
                    ),
                    2,
                ),

                "violation_rate": round(
                    float(
                        row[
                            "violation_rate"
                        ]
                    ),
                    6,
                ),

                "kupiec_pvalue": round(
                    float(
                        row[
                            "kupiec_pvalue"
                        ]
                    ),
                    6,
                ),

                "coverage_accepted_5pct": bool(
                    row[
                        "coverage_accepted_5pct"
                    ]
                ),
            }
        )

    return {

        "method": (
            "Kupiec unconditional "
            "coverage test"
        ),

        "significance_level": 0.05,

        "results": results,

        "warning": (
            "A rejected coverage test means "
            "the VaR model is not correctly "
            "calibrated at that confidence "
            "level over the test period."
        ),
    }


# ============================================================
# TOOL 4
# STRESS ANALYSIS
# ============================================================

def run_stress_analysis(
    scenario: str | None = None,
) -> dict[str, Any]:
    """
    Return stress-test results.

    Parameters
    ----------
    scenario:
        Optional scenario name.

        Available:

        MARKET_CORRECTION
        RATE_SHOCK
        BANKING_STRESS
        SEVERE_CRASH

        If None, returns all scenarios.
    """

    check_file(
        STRESS_RESULTS_FILE
    )

    check_file(
        STRESS_CONTRIBUTIONS_FILE
    )

    results = pd.read_csv(
        STRESS_RESULTS_FILE
    )

    contributions = pd.read_csv(
        STRESS_CONTRIBUTIONS_FILE
    )

    available_scenarios = (
        results[
            "scenario"
        ]
        .tolist()
    )

    if scenario is not None:

        scenario = (
            scenario
            .strip()
            .upper()
        )

        if (
            scenario
            not in available_scenarios
        ):

            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available scenarios: "
                f"{available_scenarios}"
            )

        results = (
            results[
                results[
                    "scenario"
                ]
                == scenario
            ]
        )

        contributions = (
            contributions[
                contributions[
                    "scenario"
                ]
                == scenario
            ]
        )

    output = []

    for _, row in results.iterrows():

        scenario_name = (
            row[
                "scenario"
            ]
        )

        scenario_contributions = (
            contributions[
                contributions[
                    "scenario"
                ]
                == scenario_name
            ]
        )

        asset_details = []

        for _, contribution in (
            scenario_contributions
            .iterrows()
        ):

            asset_details.append(
                {

                    "asset": (
                        contribution[
                            "asset"
                        ]
                    ),

                    "portfolio_weight": round(
                        float(
                            contribution[
                                "portfolio_weight"
                            ]
                        ),
                        4,
                    ),

                    "asset_shock_pct": round(
                        float(
                            contribution[
                                "asset_shock_pct"
                            ]
                        ),
                        2,
                    ),

                    "portfolio_loss_contribution_pct":
                        round(
                            float(
                                contribution[
                                    "loss_contribution_pct"
                                ]
                            ),
                            4,
                        ),
                }
            )

        output.append(
            {

                "scenario": (
                    scenario_name
                ),

                "description": (
                    row[
                        "description"
                    ]
                ),

                "portfolio_loss_pct": round(
                    float(
                        row[
                            "portfolio_stress_loss_pct"
                        ]
                    ),
                    4,
                ),

                "volatility_multiplier": round(
                    float(
                        row[
                            "volatility_multiplier"
                        ]
                    ),
                    2,
                ),

                "loss_multiple_of_var95": round(
                    float(
                        row[
                            "stress_loss_vs_var95"
                        ]
                    ),
                    3,
                ),

                "loss_multiple_of_var99": round(
                    float(
                        row[
                            "stress_loss_vs_var99"
                        ]
                    ),
                    3,
                ),

                "loss_multiple_of_es99": round(
                    float(
                        row[
                            "stress_loss_vs_es99"
                        ]
                    ),
                    3,
                ),

                "asset_contributions":
                    asset_details,
            }
        )

    return {

        "stress_tests_are_forecasts":
            False,

        "available_scenarios":
            available_scenarios,

        "results":
            output,
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

FINANCIAL_TOOLS = {

    "get_current_risk":
        get_current_risk,

    "compare_volatility_models":
        compare_volatility_models,

    "get_var_backtest":
        get_var_backtest,

    "run_stress_analysis":
        run_stress_analysis,
}


# ============================================================
# LOCAL TEST
# ============================================================

def main() -> None:
    """
    Test all financial tools locally.

    No LLM is used at this stage.
    """

    print(
        "\n========================================"
    )
    print(
        "TOOL 1 - CURRENT RISK"
    )
    print(
        "========================================"
    )

    print(
        get_current_risk()
    )


    print(
        "\n========================================"
    )
    print(
        "TOOL 2 - MODEL COMPARISON"
    )
    print(
        "========================================"
    )

    comparison = (
        compare_volatility_models()
    )

    print(
        f"Test period: "
        f"{comparison['test_period']}"
    )

    print(
        f"Assets: "
        f"{len(comparison['assets'])}"
    )


    print(
        "\n========================================"
    )
    print(
        "TOOL 3 - VAR BACKTEST"
    )
    print(
        "========================================"
    )

    backtest = (
        get_var_backtest()
    )

    print(
        f"Backtest rows: "
        f"{len(backtest['results'])}"
    )


    print(
        "\n========================================"
    )
    print(
        "TOOL 4 - STRESS TEST"
    )
    print(
        "========================================"
    )

    stress = (
        run_stress_analysis(
            "SEVERE_CRASH"
        )
    )

    severe = (
        stress[
            "results"
        ][0]
    )

    print(
        f"Scenario: "
        f"{severe['scenario']}"
    )

    print(
        f"Portfolio loss: "
        f"{severe['portfolio_loss_pct']}%"
    )


    print(
        "\n========================================"
    )
    print(
        "FINANCIAL TOOL REGISTRY"
    )
    print(
        "========================================"
    )

    for tool_name in (
        FINANCIAL_TOOLS.keys()
    ):

        print(
            f"- {tool_name}"
        )


    print(
        "\nFinancial agent tools "
        "validated successfully."
    )


if __name__ == "__main__":
    main()