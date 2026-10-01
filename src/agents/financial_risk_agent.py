import ast
import json
import time
from typing import Any

from ollama import chat, ChatResponse

from financial_tools import (
    get_current_risk,
    compare_volatility_models,
    get_var_backtest,
    run_stress_analysis,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "llama3.2:3b"

MAX_TOOL_ROUNDS = 6

KEEP_ALIVE = "10m"

MODEL_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 500,
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Financial Risk Agent.

You analyse an academic financial portfolio composed of:
- BNP Paribas
- AXA
- Societe Generale

You have quantitative Python tools based on:
- GARCH
- LSTM
- Hybrid GARCH-LSTM
- Value at Risk
- Expected Shortfall
- VaR backtesting
- stress testing

STRICT TOOL RULES:

1. get_current_risk takes NO arguments.

2. compare_volatility_models takes NO arguments.

3. get_var_backtest takes NO arguments.
   NEVER pass a scenario to get_var_backtest.

4. run_stress_analysis is the ONLY tool
   that may receive a scenario argument.

5. Valid stress scenarios are:
   - MARKET_CORRECTION
   - RATE_SHOCK
   - BANKING_STRESS
   - SEVERE_CRASH

GENERAL RULES:

6. Use tools whenever quantitative information is required.

7. Never invent or rescale numerical values.

8. Values ending in "_pct" are already percentages.

9. Distinguish:
   - current forecasts
   - historical backtests
   - hypothetical stress scenarios

10. Model metrics:
    - lower MSE is better
    - lower MAE is better
    - lower QLIKE is better
    - higher correlation is better

11. Kupiec test:
    - p-value < 0.05:
      coverage is rejected at 5%
    - p-value >= 0.05:
      coverage is not rejected at 5%

12. Stress scenarios are hypothetical.
    They are NOT forecasts.

13. Never provide buy or sell recommendations.

14. Answer in French when the user speaks French.

15. Never change a number returned by a tool.

16. Do not claim that one volatility model is
    globally superior based on one metric.

17. Keep the financial terminology unchanged:
    VaR means Value at Risk.
    ES means Expected Shortfall.

18. For a question with one clear intent, call exactly ONE tool.

19. Do not call additional tools unless the user explicitly asks
    for a combined or complete analysis requiring multiple tools.

20. For questions about VaR calibration, VaR violations,
    Kupiec tests or VaR backtesting, call ONLY get_var_backtest.

21. Be concise and professional.
"""


# ============================================================
# EXPLICIT TOOL SCHEMAS
# ============================================================

TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "get_current_risk",
            "description": (
                "Return the latest quantitative risk forecast "
                "for the portfolio. This function takes NO arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "compare_volatility_models",
            "description": (
                "Compare GARCH, LSTM and HYBRID volatility "
                "forecasting models using MSE, MAE, QLIKE "
                "and correlation. This function takes NO arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_var_backtest",
            "description": (
                "Return historical Value at Risk backtesting "
                "results using the Kupiec unconditional coverage test. "
                "This function takes NO arguments. "
                "NEVER provide a scenario."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "run_stress_analysis",
            "description": (
                "Return hypothetical portfolio stress-test results. "
                "A scenario may optionally be specified. "
                "This is the ONLY function accepting a scenario."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "enum": [
                            "MARKET_CORRECTION",
                            "RATE_SHOCK",
                            "BANKING_STRESS",
                            "SEVERE_CRASH",
                        ],
                        "description": (
                            "Optional hypothetical stress scenario."
                        ),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# FUNCTION REGISTRY
# ============================================================

AVAILABLE_FUNCTIONS = {
    "get_current_risk": get_current_risk,
    "compare_volatility_models": compare_volatility_models,
    "get_var_backtest": get_var_backtest,
    "run_stress_analysis": run_stress_analysis,
}


NO_ARGUMENT_TOOLS = {
    "get_current_risk",
    "compare_volatility_models",
    "get_var_backtest",
}


VALID_STRESS_SCENARIOS = {
    "MARKET_CORRECTION",
    "RATE_SHOCK",
    "BANKING_STRESS",
    "SEVERE_CRASH",
}


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_tool_result(
    result: Any,
) -> str:

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


# ============================================================
# ARGUMENT GUARDRAIL
# ============================================================

def sanitize_arguments(
    tool_name: str,
    arguments: dict | None,
) -> dict:

    if arguments is None:
        arguments = {}

    if tool_name in NO_ARGUMENT_TOOLS:

        if arguments:

            print(
                "[Guardrail] Invalid arguments removed "
                f"from {tool_name}: {arguments}"
            )

        return {}

    if tool_name == "run_stress_analysis":

        scenario = arguments.get(
            "scenario"
        )

        if scenario is None:
            return {}

        if not isinstance(
            scenario,
            str,
        ):
            raise ValueError(
                "Stress scenario must be a string."
            )

        cleaned = (
            scenario
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # Normalize requests meaning "all stress scenarios"
        # ----------------------------------------------------

        if cleaned in {
            "",
            "NONE",
            "NULL",
            "ALL",
        }:
            return {}

        # Llama may sometimes return all scenarios as a
        # string representation of a Python list:
        #
        # "['MARKET_CORRECTION', 'RATE_SHOCK', ...]"
        #
        # Convert this safely before executing the tool.

        if (
            cleaned.startswith("[")
            and
            cleaned.endswith("]")
        ):

            try:

                parsed = ast.literal_eval(
                    scenario
                )

                if isinstance(
                    parsed,
                    (list, tuple, set),
                ):

                    normalized = {
                        str(item)
                        .strip()
                        .upper()
                        for item in parsed
                    }

                    # All available scenarios requested
                    if normalized == VALID_STRESS_SCENARIOS:

                        print(
                            "[Guardrail] Stress scenario list "
                            "normalized to ALL scenarios."
                        )

                        return {}

                    # One valid scenario wrapped in a list
                    if (
                        len(normalized) == 1
                        and
                        next(iter(normalized))
                        in VALID_STRESS_SCENARIOS
                    ):

                        return {
                            "scenario":
                                next(iter(normalized))
                        }

            except (
                ValueError,
                SyntaxError,
            ):

                pass

        if cleaned not in (
            VALID_STRESS_SCENARIOS
        ):
            raise ValueError(
                f"Invalid stress scenario: {scenario}"
            )

        return {
            "scenario": cleaned
        }

    return arguments


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict | None,
) -> tuple[str, Any]:

    function = AVAILABLE_FUNCTIONS.get(
        tool_name
    )

    if function is None:

        error = {
            "error":
                f"Unknown tool: {tool_name}"
        }

        return (
            serialize_tool_result(error),
            error,
        )

    try:

        clean_arguments = (
            sanitize_arguments(
                tool_name,
                arguments,
            )
        )

        raw_result = function(
            **clean_arguments
        )

        return (
            serialize_tool_result(
                raw_result
            ),
            raw_result,
        )

    except Exception as error:

        result = {
            "tool": tool_name,
            "error": str(error),
        }

        return (
            serialize_tool_result(result),
            result,
        )


# ============================================================
# CURRENT RISK FORMATTER
# ============================================================

def format_current_risk_answer(
    result: dict,
) -> str:

    date = result["date"]
    model = result["model"]

    daily_volatility = result[
        "daily_volatility_pct"
    ]

    annualized_volatility = result[
        "annualized_volatility_pct"
    ]

    var_95 = result[
        "var_95_pct"
    ]

    es_95 = result[
        "expected_shortfall_95_pct"
    ]

    var_99 = result[
        "var_99_pct"
    ]

    es_99 = result[
        "expected_shortfall_99_pct"
    ]

    return (
        f"Au {date}, selon le modèle {model} :\n\n"

        f"- Volatilité quotidienne : "
        f"{daily_volatility:.4f} %\n"

        f"- Volatilité annualisée : "
        f"{annualized_volatility:.4f} %\n"

        f"- VaR 95 % à 1 jour : "
        f"{var_95:.4f} %\n"

        f"- Expected Shortfall 95 % : "
        f"{es_95:.4f} %\n"

        f"- VaR 99 % à 1 jour : "
        f"{var_99:.4f} %\n"

        f"- Expected Shortfall 99 % : "
        f"{es_99:.4f} %\n\n"

        f"Une VaR 95 % de "
        f"{var_95:.4f} % signifie que, "
        f"sous les hypothèses du modèle, "
        f"environ 5 % des observations peuvent "
        f"dépasser ce seuil de perte."
    )


# ============================================================
# MODEL COMPARISON FORMATTER
# ============================================================

def format_model_comparison_answer(
    result: dict,
) -> str:
    """
    Deterministic comparison of GARCH,
    LSTM and HYBRID.

    Winners are calculated directly
    from the quantitative tool output.
    """

    assets = result.get(
        "assets",
        [],
    )

    test_period = result.get(
        "test_period",
        "unknown",
    )

    if not assets:

        return (
            "Aucun résultat de comparaison "
            "des modèles n'est disponible."
        )

    models = [
        "GARCH",
        "LSTM",
        "HYBRID",
    ]

    metrics = [
        "mse",
        "mae",
        "qlike",
        "correlation",
    ]

    winner_counts = {
        metric: {
            model: 0
            for model in models
        }
        for metric in metrics
    }

    sections = []

    # ========================================================
    # ASSET-BY-ASSET RESULTS
    # ========================================================

    for asset_data in assets:

        asset = asset_data[
            "asset"
        ]

        lines = [
            f"{asset} :"
        ]

        # ----------------------------------------------------
        # Display numerical metrics
        # ----------------------------------------------------

        for model in models:

            data = asset_data[
                model
            ]

            lines.append(
                f"- {model} : "
                f"MSE={data['mse']:.6f}, "
                f"MAE={data['mae']:.6f}, "
                f"QLIKE={data['qlike']:.6f}, "
                f"corrélation="
                f"{data['correlation']:.6f}"
            )

        # ----------------------------------------------------
        # Determine winners
        # ----------------------------------------------------

        mse_winner = min(
            models,
            key=lambda model:
                asset_data[
                    model
                ]["mse"],
        )

        mae_winner = min(
            models,
            key=lambda model:
                asset_data[
                    model
                ]["mae"],
        )

        qlike_winner = min(
            models,
            key=lambda model:
                asset_data[
                    model
                ]["qlike"],
        )

        correlation_winner = max(
            models,
            key=lambda model:
                asset_data[
                    model
                ]["correlation"],
        )

        winner_counts[
            "mse"
        ][mse_winner] += 1

        winner_counts[
            "mae"
        ][mae_winner] += 1

        winner_counts[
            "qlike"
        ][qlike_winner] += 1

        winner_counts[
            "correlation"
        ][correlation_winner] += 1

        lines.extend(
            [
                "",
                "Meilleure valeur par métrique :",
                f"- MSE : {mse_winner}",
                f"- MAE : {mae_winner}",
                f"- QLIKE : {qlike_winner}",
                f"- Corrélation : "
                f"{correlation_winner}",
            ]
        )

        sections.append(
            "\n".join(lines)
        )

    # ========================================================
    # GLOBAL COUNTS
    # ========================================================

    total_assets = len(
        assets
    )

    summary_lines = [
        (
            f"Résumé sur les "
            f"{total_assets} actifs :"
        ),
        "",
    ]

    pretty_metric_names = {
        "mse": "MSE",
        "mae": "MAE",
        "qlike": "QLIKE",
        "correlation": "Corrélation",
    }

    for metric in metrics:

        counts = winner_counts[
            metric
        ]

        summary_lines.append(
            f"- "
            f"{pretty_metric_names[metric]} : "
            f"GARCH {counts['GARCH']}, "
            f"LSTM {counts['LSTM']}, "
            f"HYBRID {counts['HYBRID']} "
            f"victoire(s)."
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    interpretation = (
        "Interprétation : les modèles présentent "
        "des avantages différents selon la métrique. "
        "Le MSE, le MAE, le QLIKE et la corrélation "
        "ne mesurent pas exactement la même propriété. "
        "Il serait donc incorrect de déclarer un modèle "
        "globalement supérieur uniquement à partir "
        "d'une seule métrique."
    )

    return (
        "Comparaison des modèles de volatilité\n"
        f"Période de test : {test_period}\n\n"

        + "\n\n".join(
            sections
        )

        + "\n\n"

        + "\n".join(
            summary_lines
        )

        + "\n\n"

        + interpretation
    )


# ============================================================
# BACKTEST HELPERS
# ============================================================

def confidence_to_pct(
    value: float,
) -> float:

    value = float(value)

    if value <= 1.0:
        return value * 100.0

    return value


def rate_to_pct(
    value: float,
) -> float:

    value = float(value)

    if value <= 1.0:
        return value * 100.0

    return value


# ============================================================
# VAR BACKTEST FORMATTER
# ============================================================

def format_var_backtest_answer(
    result: dict,
) -> str:

    rows = result.get(
        "results",
        [],
    )

    if not rows:

        return (
            "Aucun résultat de backtest "
            "VaR n'est disponible."
        )

    lines = []
    accepted_cases = []

    for row in rows:

        model = str(
            row["model"]
        )

        confidence = (
            confidence_to_pct(
                row[
                    "confidence_level"
                ]
            )
        )

        observations = int(
            row["observations"]
        )

        violations = int(
            row["violations"]
        )

        expected = float(
            row[
                "expected_violations"
            ]
        )

        violation_rate = (
            rate_to_pct(
                row[
                    "violation_rate"
                ]
            )
        )

        pvalue = float(
            row[
                "kupiec_pvalue"
            ]
        )

        accepted = bool(
            row[
                "coverage_accepted_5pct"
            ]
        )

        status = (
            "non rejetée"
            if accepted
            else "rejetée"
        )

        lines.append(
            f"- {model} — VaR "
            f"{confidence:.0f} % : "
            f"{violations} violations "
            f"sur {observations} "
            f"(attendu ≈ {expected:.2f}), "
            f"taux observé "
            f"{violation_rate:.2f} %, "
            f"p-value Kupiec = "
            f"{pvalue:.6f} → "
            f"couverture {status} à 5 %."
        )

        if accepted:

            accepted_cases.append(
                f"{model} "
                f"{confidence:.0f} %"
            )

    if len(
        accepted_cases
    ) == 1:

        conclusion = (
            "Sur ce test, seule la configuration "
            f"{accepted_cases[0]} n'est pas rejetée "
            "au seuil de 5 %."
        )

    elif len(
        accepted_cases
    ) == 0:

        conclusion = (
            "Toutes les configurations sont "
            "rejetées par le test de Kupiec "
            "au seuil de 5 %."
        )

    else:

        conclusion = (
            "Configurations non rejetées au "
            "seuil de 5 % : "
            + ", ".join(
                accepted_cases
            )
            + "."
        )

    return (
        "Backtest historique de la VaR "
        "— test de couverture de Kupiec :\n\n"

        + "\n".join(
            lines
        )

        + "\n\n"

        + conclusion

        + "\n\n"

        + (
            "Attention : ne pas rejeter la couverture "
            "avec le test de Kupiec ne prouve pas une "
            "calibration parfaite. Ce test vérifie "
            "principalement la fréquence des violations."
        )
    )


# ============================================================
# STRESS TEST FORMATTER
# ============================================================

def format_stress_answer(
    result: dict,
) -> str:

    rows = result.get(
        "results",
        [],
    )

    if not rows:

        return (
            "Aucun résultat de stress test "
            "n'est disponible."
        )

    sections = []

    for row in rows:

        scenario = str(
            row["scenario"]
        )

        description = str(
            row.get(
                "description",
                "",
            )
        )

        portfolio_loss = float(
            row[
                "portfolio_loss_pct"
            ]
        )

        volatility_multiplier = float(
            row[
                "volatility_multiplier"
            ]
        )

        loss_var95 = float(
            row[
                "loss_multiple_of_var95"
            ]
        )

        loss_var99 = float(
            row[
                "loss_multiple_of_var99"
            ]
        )

        loss_es99 = float(
            row[
                "loss_multiple_of_es99"
            ]
        )

        contributions = row.get(
            "asset_contributions",
            [],
        )

        lines = [
            f"Scénario : {scenario}",
        ]

        if description:

            lines.append(
                f"Description : "
                f"{description}"
            )

        lines.extend(
            [
                "",
                (
                    "- Perte hypothétique "
                    "du portefeuille : "
                    f"{portfolio_loss:.4f} %"
                ),
                (
                    "- Multiplicateur de volatilité "
                    "du scénario : "
                    f"{volatility_multiplier:.2f}×"
                ),
                "",
                "Chocs par actif :",
            ]
        )

        for item in contributions:

            asset = str(
                item["asset"]
            )

            weight = float(
                item[
                    "portfolio_weight"
                ]
            )

            shock = float(
                item[
                    "asset_shock_pct"
                ]
            )

            contribution = float(
                item[
                    "portfolio_loss_contribution_pct"
                ]
            )

            if weight <= 1.0:

                weight_pct = (
                    weight * 100.0
                )

            else:

                weight_pct = weight

            lines.append(
                f"- {asset} : "
                f"poids {weight_pct:.2f} %, "
                f"choc {shock:.2f} %, "
                f"contribution à la perte "
                f"{contribution:.2f} %."
            )

        lines.extend(
            [
                "",
                (
                    "Comparaison aux "
                    "mesures de risque :"
                ),
                (
                    "- Perte du scénario / "
                    f"VaR 95 % : "
                    f"{loss_var95:.3f}×"
                ),
                (
                    "- Perte du scénario / "
                    f"VaR 99 % : "
                    f"{loss_var99:.3f}×"
                ),
                (
                    "- Perte du scénario / "
                    f"ES 99 % : "
                    f"{loss_es99:.3f}×"
                ),
                "",
                (
                    "Ce stress test est un scénario "
                    "hypothétique déterministe. "
                    "Il ne constitue pas une "
                    "prévision de marché."
                ),
            ]
        )

        sections.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# MODEL CALL
# ============================================================

def call_model(
    conversation: list,
) -> ChatResponse:

    start_time = (
        time.perf_counter()
    )

    print(
        "\n[Llama] Generating...",
        flush=True,
    )

    response: ChatResponse = chat(

        model=MODEL,

        messages=conversation,

        tools=TOOLS,

        keep_alive=KEEP_ALIVE,

        options=MODEL_OPTIONS,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"[Llama] Done in "
        f"{elapsed:.2f} s",
        flush=True,
    )

    return response


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(
    user_message: str,
    conversation: list,
) -> tuple[str, list]:

    conversation.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    for round_number in range(
        1,
        MAX_TOOL_ROUNDS + 1,
    ):

        response = call_model(
            conversation
        )

        assistant_message = (
            response.message
        )

        conversation.append(
            assistant_message
        )

        tool_calls = (
            assistant_message.tool_calls
            or []
        )

        # ====================================================
        # SINGLE-INTENT VAR ORCHESTRATION GUARDRAIL
        # ====================================================

        message_lower = user_message.lower()

        var_keywords = (
            "var",
            "backtest",
            "kupiec",
            "violation",
            "violations",
            "calibr",
        )

        multi_intent_keywords = (
            "analyse complète",
            "analyse globale",
            "tous les outils",
            "analyse tout",
        )

        is_single_var_request = (
            any(
                keyword in message_lower
                for keyword in var_keywords
            )
            and
            not any(
                keyword in message_lower
                for keyword in multi_intent_keywords
            )
        )

        if (
            len(tool_calls) > 1
            and
            is_single_var_request
            and
            tool_calls[0].function.name
            == "get_var_backtest"
        ):
            print(
                "[Guardrail] Multiple tool calls reduced "
                "to get_var_backtest."
            )

            tool_calls = [
                tool_calls[0]
            ]

        # ====================================================
        # NORMAL TEXT ANSWER
        # ====================================================

        if not tool_calls:

            final_answer = (
                assistant_message.content
                or ""
            ).strip()

            if not final_answer:

                final_answer = (
                    "Aucune réponse textuelle "
                    "n'a été générée."
                )

            return (
                final_answer,
                conversation,
            )

        # ====================================================
        # TOOL CALLS
        # ====================================================

        print(
            "\n========================================"
        )

        print(
            f"AGENT ROUND {round_number}"
        )

        print(
            "========================================"
        )

        executed_tools = []

        for tool_call in tool_calls:

            tool_name = (
                tool_call
                .function
                .name
            )

            arguments = (
                tool_call
                .function
                .arguments
                or {}
            )

            print(
                f"Tool selected : "
                f"{tool_name}"
            )

            print(
                f"Arguments     : "
                f"{arguments}"
            )

            (
                serialized_result,
                raw_result,
            ) = execute_tool(
                tool_name,
                arguments,
            )

            print(
                "Tool executed successfully."
            )

            executed_tools.append(
                {
                    "name": tool_name,
                    "result": raw_result,
                }
            )

            conversation.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content":
                        serialized_result,
                }
            )

        # ====================================================
        # CURRENT RISK GUARDRAIL
        # ====================================================

        if (
            len(executed_tools) == 1
            and
            executed_tools[0]["name"]
            == "get_current_risk"
        ):

            result = (
                executed_tools[0]
                ["result"]
            )

            final_answer = (
                format_current_risk_answer(
                    result
                )
            )

            conversation.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                }
            )

            print(
                "[Guardrail] Current-risk "
                "numbers rendered by Python."
            )

            return (
                final_answer,
                conversation,
            )

        # ====================================================
        # MODEL COMPARISON GUARDRAIL
        # ====================================================

        if (
            len(executed_tools) == 1
            and
            executed_tools[0]["name"]
            == "compare_volatility_models"
        ):

            result = (
                executed_tools[0]
                ["result"]
            )

            if "error" not in result:

                final_answer = (
                    format_model_comparison_answer(
                        result
                    )
                )

                conversation.append(
                    {
                        "role": "assistant",
                        "content":
                            final_answer,
                    }
                )

                print(
                    "[Guardrail] Model comparison "
                    "rendered by Python."
                )

                return (
                    final_answer,
                    conversation,
                )

        # ====================================================
        # VAR BACKTEST GUARDRAIL
        # ====================================================

        if (
            len(executed_tools) == 1
            and
            executed_tools[0]["name"]
            == "get_var_backtest"
        ):

            result = (
                executed_tools[0]
                ["result"]
            )

            if "error" not in result:

                final_answer = (
                    format_var_backtest_answer(
                        result
                    )
                )

                conversation.append(
                    {
                        "role": "assistant",
                        "content":
                            final_answer,
                    }
                )

                print(
                    "[Guardrail] VaR backtest "
                    "rendered by Python."
                )

                return (
                    final_answer,
                    conversation,
                )

        # ====================================================
        # STRESS TEST GUARDRAIL
        # ====================================================

        if (
            len(executed_tools) == 1
            and
            executed_tools[0]["name"]
            == "run_stress_analysis"
        ):

            result = (
                executed_tools[0]
                ["result"]
            )

            if "error" not in result:

                final_answer = (
                    format_stress_answer(
                        result
                    )
                )

                conversation.append(
                    {
                        "role": "assistant",
                        "content":
                            final_answer,
                    }
                )

                print(
                    "[Guardrail] Stress-test "
                    "numbers rendered by Python."
                )

                return (
                    final_answer,
                    conversation,
                )

    raise RuntimeError(
        "Maximum number of agent "
        "tool rounds reached."
    )


# ============================================================
# INTRODUCTION
# ============================================================

def print_intro() -> None:

    print(
        "\n========================================"
    )

    print(
        "FINANCIAL RISK AGENT"
    )

    print(
        "========================================"
    )

    print(
        f"Local model : {MODEL}"
    )

    print(
        "Engine      : Ollama"
    )

    print(
        "API cost    : 0 €"
    )

    print(
        "Guardrails  : "
        "quantitative + arguments"
    )

    print(
        "\nAvailable quantitative tools:"
    )

    for tool_name in (
        AVAILABLE_FUNCTIONS.keys()
    ):

        print(
            f"- {tool_name}"
        )

    print(
        "\nExamples:"
    )

    print(
        "- Quel est le risque actuel "
        "de mon portefeuille ?"
    )

    print(
        "- Compare GARCH, LSTM et Hybrid."
    )

    print(
        "- Est-ce que notre VaR "
        "est bien calibrée ?"
    )

    print(
        "- Analyse le scénario "
        "SEVERE_CRASH."
    )

    print(
        "- Fais-moi une analyse complète "
        "du risque."
    )

    print(
        "\nCommands:"
    )

    print(
        "/clear  -> reset conversation"
    )

    print(
        "/exit   -> quit"
    )

    print(
        "========================================\n"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_intro()

    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    while True:

        try:

            user_input = input(
                "You > "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nFinancial Risk Agent stopped."
            )

            break

        except EOFError:

            print(
                "\nFinancial Risk Agent stopped."
            )

            break

        if not user_input:

            continue

        if user_input.lower() in {
            "/exit",
            "/quit",
            "exit",
            "quit",
        }:

            print(
                "Financial Risk Agent stopped."
            )

            break

        if user_input.lower() == "/clear":

            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print(
                "Conversation cleared.\n"
            )

            continue

        try:

            (
                answer,
                conversation,
            ) = run_agent(
                user_input,
                conversation,
            )

            print(
                "\n========================================"
            )

            print(
                "AGENT ANSWER"
            )

            print(
                "========================================"
            )

            print(
                answer
            )

            print()

        except KeyboardInterrupt:

            print(
                "\nGeneration interrupted.\n"
            )

        except Exception as error:

            print(
                "\n========================================"
            )

            print(
                "AGENT ERROR"
            )

            print(
                "========================================"
            )

            print(
                str(error)
            )

            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()