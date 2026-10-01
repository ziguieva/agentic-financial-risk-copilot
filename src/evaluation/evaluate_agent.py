import json
import sys
import time
from pathlib import Path
from statistics import mean

import pandas as pd
from ollama import chat


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

AGENTS_DIR = (
    ROOT_DIR
    / "src"
    / "agents"
)

TEST_FILE = (
    ROOT_DIR
    / "tests"
    / "agent_evaluation_cases.json"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "agent_evaluation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(AGENTS_DIR),
)


# ============================================================
# IMPORT REAL AGENT CONFIGURATION
# ============================================================

from financial_risk_agent import (  # noqa: E402
    MODEL,
    TOOLS,
    SYSTEM_PROMPT,
    KEEP_ALIVE,
    MODEL_OPTIONS,
    AVAILABLE_FUNCTIONS,
    sanitize_arguments,
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_cases() -> list[dict]:

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        cases = json.load(file)

    if not isinstance(cases, list):

        raise ValueError(
            "Evaluation dataset must be a JSON list."
        )

    return cases


# ============================================================
# NORMALIZE ARGUMENTS
# ============================================================

def normalize_arguments(
    arguments,
) -> dict:

    if arguments is None:

        return {}

    if isinstance(
        arguments,
        dict,
    ):

        return arguments

    if isinstance(
        arguments,
        str,
    ):

        text = arguments.strip()

        if not text:

            return {}

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except json.JSONDecodeError:

            pass

    return {
        "_raw": str(arguments)
    }


# ============================================================
# JSON FOR CSV
# ============================================================

def json_text(
    value,
) -> str:

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# ORCHESTRATION GUARDRAIL
# ============================================================

def apply_orchestration_guardrail(
    question: str,
    tool_calls: list,
) -> list:
    """
    Reproduce the orchestration protection
    used by the final Financial Risk Agent.

    For a single-intent VaR/backtest question,
    extra tool calls are removed and only
    get_var_backtest is retained.
    """

    guarded_calls = list(
        tool_calls
    )

    message_lower = (
        question.lower()
    )

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
        len(guarded_calls) > 1
        and
        is_single_var_request
        and
        guarded_calls[0]
        .function
        .name
        == "get_var_backtest"
    ):

        guarded_calls = [
            guarded_calls[0]
        ]

    return guarded_calls


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:

        return 0.0

    values = sorted(
        values
    )

    position = (
        percentile_value
        / 100.0
        * (
            len(values) - 1
        )
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        +
        (
            values[upper]
            - values[lower]
        )
        * fraction
    )


# ============================================================
# EVALUATE ONE CASE
# ============================================================

def evaluate_case(
    case: dict,
) -> dict:

    question = case[
        "question"
    ]

    expected_tool = case[
        "expected_tool"
    ]

    expected_arguments = case.get(
        "expected_arguments",
        {},
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    start = time.perf_counter()

    try:

        response = chat(

            model=MODEL,

            messages=messages,

            tools=TOOLS,

            keep_alive=KEEP_ALIVE,

            options=MODEL_OPTIONS,
        )

        latency = (
            time.perf_counter()
            - start
        )

        raw_tool_calls = (
            response
            .message
            .tool_calls
            or []
        )

        raw_call_count = len(
            raw_tool_calls
        )

        # ====================================================
        # NO TOOL CALL
        # ====================================================

        if raw_call_count == 0:

            return {

                "id":
                    case["id"],

                "category":
                    case["category"],

                "question":
                    question,

                "expected_tool":
                    expected_tool,

                "expected_arguments":
                    json_text(
                        expected_arguments
                    ),

                "first_predicted_tool":
                    None,

                "first_raw_arguments":
                    "{}",

                "first_guarded_arguments":
                    "{}",

                "all_raw_tools":
                    "[]",

                "all_guarded_tools":
                    "[]",

                "all_raw_arguments":
                    "[]",

                "raw_tool_call_count":
                    0,

                "guarded_tool_call_count":
                    0,

                "top1_tool_correct":
                    False,

                "raw_single_call_compliant":
                    False,

                "guarded_single_call_compliant":
                    False,

                "top1_raw_arguments_correct":
                    False,

                "top1_guarded_arguments_correct":
                    False,

                "raw_strict_call_correct":
                    False,

                "final_execution_correct":
                    False,

                "top1_valid_call":
                    False,

                "latency_seconds":
                    round(
                        latency,
                        4,
                    ),

                "error":
                    "No tool call",
            }

        # ====================================================
        # RAW LLM OUTPUT
        # ====================================================

        raw_tools = []

        raw_arguments_list = []

        for call in raw_tool_calls:

            tool_name = (
                call
                .function
                .name
            )

            arguments = normalize_arguments(
                call
                .function
                .arguments
            )

            raw_tools.append(
                tool_name
            )

            raw_arguments_list.append(
                arguments
            )

        first_tool = (
            raw_tools[0]
        )

        first_raw_arguments = (
            raw_arguments_list[0]
        )

        # ====================================================
        # ORCHESTRATION GUARDRAIL
        # ====================================================

        guarded_tool_calls = (
            apply_orchestration_guardrail(
                question,
                raw_tool_calls,
            )
        )

        guarded_call_count = len(
            guarded_tool_calls
        )

        guarded_tools = [
            call.function.name
            for call
            in guarded_tool_calls
        ]

        # ====================================================
        # ARGUMENT GUARDRAIL
        # ====================================================

        first_guardrail_error = None

        try:

            first_guarded_arguments = (
                sanitize_arguments(
                    first_tool,
                    first_raw_arguments,
                )
            )

        except Exception as error:

            first_guarded_arguments = {}

            first_guardrail_error = str(
                error
            )

        # ====================================================
        # METRICS
        # ====================================================

        top1_tool_correct = (
            first_tool
            == expected_tool
        )

        raw_single_call_compliant = (
            raw_call_count
            == 1
        )

        guarded_single_call_compliant = (
            guarded_call_count
            == 1
        )

        top1_raw_arguments_correct = (
            top1_tool_correct
            and
            first_raw_arguments
            == expected_arguments
        )

        top1_guarded_arguments_correct = (
            top1_tool_correct
            and
            first_guardrail_error is None
            and
            first_guarded_arguments
            == expected_arguments
        )

        top1_known_tool = (
            first_tool
            in AVAILABLE_FUNCTIONS
        )

        top1_valid_call = (
            top1_known_tool
            and
            first_guardrail_error is None
        )

        # ----------------------------------------------------
        # RAW STRICT ACCURACY
        # ----------------------------------------------------

        raw_strict_call_correct = (
            top1_tool_correct
            and
            top1_raw_arguments_correct
            and
            raw_single_call_compliant
        )

        # ----------------------------------------------------
        # FINAL AGENT EXECUTION ACCURACY
        # ----------------------------------------------------

        final_execution_correct = (
            top1_tool_correct
            and
            top1_guarded_arguments_correct
            and
            guarded_single_call_compliant
            and
            top1_valid_call
        )

        # ====================================================
        # ERROR / NOTE
        # ====================================================

        notes = []

        if not top1_tool_correct:

            notes.append(
                "Wrong first tool"
            )

        if not raw_single_call_compliant:

            notes.append(
                f"Raw multiple calls: "
                f"{raw_call_count}"
            )

        if (
            raw_call_count
            != guarded_call_count
        ):

            notes.append(
                f"Orchestration reduced calls "
                f"{raw_call_count} -> "
                f"{guarded_call_count}"
            )

        if not top1_raw_arguments_correct:

            notes.append(
                "Raw arguments mismatch"
            )

        if (
            first_raw_arguments
            != first_guarded_arguments
        ):

            notes.append(
                "Arguments normalized by guardrail"
            )

        if (
            first_guardrail_error
            is not None
        ):

            notes.append(
                "Guardrail error: "
                + first_guardrail_error
            )

        error_text = (
            " | ".join(notes)
            if notes
            else None
        )

        return {

            "id":
                case["id"],

            "category":
                case["category"],

            "question":
                question,

            "expected_tool":
                expected_tool,

            "expected_arguments":
                json_text(
                    expected_arguments
                ),

            "first_predicted_tool":
                first_tool,

            "first_raw_arguments":
                json_text(
                    first_raw_arguments
                ),

            "first_guarded_arguments":
                json_text(
                    first_guarded_arguments
                ),

            "all_raw_tools":
                json_text(
                    raw_tools
                ),

            "all_guarded_tools":
                json_text(
                    guarded_tools
                ),

            "all_raw_arguments":
                json_text(
                    raw_arguments_list
                ),

            "raw_tool_call_count":
                raw_call_count,

            "guarded_tool_call_count":
                guarded_call_count,

            "top1_tool_correct":
                top1_tool_correct,

            "raw_single_call_compliant":
                raw_single_call_compliant,

            "guarded_single_call_compliant":
                guarded_single_call_compliant,

            "top1_raw_arguments_correct":
                top1_raw_arguments_correct,

            "top1_guarded_arguments_correct":
                top1_guarded_arguments_correct,

            "raw_strict_call_correct":
                raw_strict_call_correct,

            "final_execution_correct":
                final_execution_correct,

            "top1_valid_call":
                top1_valid_call,

            "latency_seconds":
                round(
                    latency,
                    4,
                ),

            "error":
                error_text,
        }

    except Exception as error:

        latency = (
            time.perf_counter()
            - start
        )

        return {

            "id":
                case["id"],

            "category":
                case["category"],

            "question":
                question,

            "expected_tool":
                expected_tool,

            "expected_arguments":
                json_text(
                    expected_arguments
                ),

            "first_predicted_tool":
                None,

            "first_raw_arguments":
                "{}",

            "first_guarded_arguments":
                "{}",

            "all_raw_tools":
                "[]",

            "all_guarded_tools":
                "[]",

            "all_raw_arguments":
                "[]",

            "raw_tool_call_count":
                0,

            "guarded_tool_call_count":
                0,

            "top1_tool_correct":
                False,

            "raw_single_call_compliant":
                False,

            "guarded_single_call_compliant":
                False,

            "top1_raw_arguments_correct":
                False,

            "top1_guarded_arguments_correct":
                False,

            "raw_strict_call_correct":
                False,

            "final_execution_correct":
                False,

            "top1_valid_call":
                False,

            "latency_seconds":
                round(
                    latency,
                    4,
                ),

            "error":
                str(error),
        }


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    dataframe: pd.DataFrame,
) -> dict:

    total = len(
        dataframe
    )

    def percentage(
        column: str,
    ) -> float:

        return (
            dataframe[
                column
            ].mean()
            * 100.0
        )

    latencies = (
        dataframe[
            "latency_seconds"
        ]
        .astype(float)
        .tolist()
    )

    category_summary = {}

    for (
        category,
        group,
    ) in dataframe.groupby(
        "category"
    ):

        category_summary[
            category
        ] = {

            "n_cases":
                int(
                    len(group)
                ),

            "top1_tool_accuracy_pct":
                round(
                    group[
                        "top1_tool_correct"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "raw_single_call_compliance_pct":
                round(
                    group[
                        "raw_single_call_compliant"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "guarded_single_call_compliance_pct":
                round(
                    group[
                        "guarded_single_call_compliant"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "raw_argument_accuracy_pct":
                round(
                    group[
                        "top1_raw_arguments_correct"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "guarded_argument_accuracy_pct":
                round(
                    group[
                        "top1_guarded_arguments_correct"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "raw_strict_accuracy_pct":
                round(
                    group[
                        "raw_strict_call_correct"
                    ].mean()
                    * 100.0,
                    2,
                ),

            "final_execution_accuracy_pct":
                round(
                    group[
                        "final_execution_correct"
                    ].mean()
                    * 100.0,
                    2,
                ),
        }

    return {

        "model":
            MODEL,

        "total_cases":
            total,

        "top1_tool_accuracy_pct":
            round(
                percentage(
                    "top1_tool_correct"
                ),
                2,
            ),

        "raw_single_call_compliance_pct":
            round(
                percentage(
                    "raw_single_call_compliant"
                ),
                2,
            ),

        "guarded_single_call_compliance_pct":
            round(
                percentage(
                    "guarded_single_call_compliant"
                ),
                2,
            ),

        "top1_raw_argument_accuracy_pct":
            round(
                percentage(
                    "top1_raw_arguments_correct"
                ),
                2,
            ),

        "top1_guarded_argument_accuracy_pct":
            round(
                percentage(
                    "top1_guarded_arguments_correct"
                ),
                2,
            ),

        "raw_strict_call_accuracy_pct":
            round(
                percentage(
                    "raw_strict_call_correct"
                ),
                2,
            ),

        "final_execution_accuracy_pct":
            round(
                percentage(
                    "final_execution_correct"
                ),
                2,
            ),

        "top1_valid_call_rate_pct":
            round(
                percentage(
                    "top1_valid_call"
                ),
                2,
            ),

        "raw_multiple_tool_call_rate_pct":
            round(
                (
                    (
                        dataframe[
                            "raw_tool_call_count"
                        ]
                        > 1
                    ).mean()
                    * 100.0
                ),
                2,
            ),

        "guarded_multiple_tool_call_rate_pct":
            round(
                (
                    (
                        dataframe[
                            "guarded_tool_call_count"
                        ]
                        > 1
                    ).mean()
                    * 100.0
                ),
                2,
            ),

        "mean_latency_seconds":
            round(
                mean(latencies),
                4,
            ),

        "p95_latency_seconds":
            round(
                percentile(
                    latencies,
                    95,
                ),
                4,
            ),

        "max_latency_seconds":
            round(
                max(latencies),
                4,
            ),

        "by_category":
            category_summary,
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary: dict,
) -> None:

    print(
        "\n========================================"
    )

    print(
        "AGENT EVALUATION SUMMARY V3"
    )

    print(
        "========================================"
    )

    print(
        f"Model                           : "
        f"{summary['model']}"
    )

    print(
        f"Test cases                      : "
        f"{summary['total_cases']}"
    )

    print()

    print(
        f"Top-1 tool accuracy             : "
        f"{summary['top1_tool_accuracy_pct']:.2f} %"
    )

    print()

    print(
        "--- RAW LLM ---"
    )

    print(
        f"Raw single-call compliance      : "
        f"{summary['raw_single_call_compliance_pct']:.2f} %"
    )

    print(
        f"Raw argument accuracy           : "
        f"{summary['top1_raw_argument_accuracy_pct']:.2f} %"
    )

    print(
        f"Raw strict call accuracy        : "
        f"{summary['raw_strict_call_accuracy_pct']:.2f} %"
    )

    print(
        f"Raw multiple-call rate          : "
        f"{summary['raw_multiple_tool_call_rate_pct']:.2f} %"
    )

    print()

    print(
        "--- AFTER GUARDRAILS ---"
    )

    print(
        f"Guarded single-call compliance  : "
        f"{summary['guarded_single_call_compliance_pct']:.2f} %"
    )

    print(
        f"Guarded argument accuracy       : "
        f"{summary['top1_guarded_argument_accuracy_pct']:.2f} %"
    )

    print(
        f"Final execution accuracy        : "
        f"{summary['final_execution_accuracy_pct']:.2f} %"
    )

    print(
        f"Guarded multiple-call rate      : "
        f"{summary['guarded_multiple_tool_call_rate_pct']:.2f} %"
    )

    print()

    print(
        "--- LATENCY ---"
    )

    print(
        f"Mean latency                    : "
        f"{summary['mean_latency_seconds']:.3f} s"
    )

    print(
        f"P95 latency                     : "
        f"{summary['p95_latency_seconds']:.3f} s"
    )

    print(
        f"Maximum latency                 : "
        f"{summary['max_latency_seconds']:.3f} s"
    )

    print(
        "\n========================================"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    cases = load_test_cases()

    print(
        "\n========================================"
    )

    print(
        "FINANCIAL RISK AGENT EVALUATION V3"
    )

    print(
        "========================================"
    )

    print(
        f"Model      : {MODEL}"
    )

    print(
        f"Test cases : {len(cases)}"
    )

    print(
        "========================================\n"
    )

    results = []

    for (
        index,
        case,
    ) in enumerate(
        cases,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{case['question']}"
        )

        result = evaluate_case(
            case
        )

        results.append(
            result
        )

        print(
            f"     First tool     : "
            f"{result['first_predicted_tool']}"
        )

        print(
            f"     Raw calls      : "
            f"{result['raw_tool_call_count']}"
        )

        print(
            f"     Guarded calls  : "
            f"{result['guarded_tool_call_count']}"
        )

        print(
            f"     Raw arguments  : "
            f"{result['first_raw_arguments']}"
        )

        print(
            f"     Guarded args   : "
            f"{result['first_guarded_arguments']}"
        )

        print(
            f"     Final correct  : "
            f"{result['final_execution_correct']}"
        )

        if result[
            "error"
        ]:

            print(
                f"     Note           : "
                f"{result['error']}"
            )

        print()

    dataframe = pd.DataFrame(
        results
    )

    detailed_path = (
        RESULTS_DIR
        / "agent_evaluation_results_v3.csv"
    )

    dataframe.to_csv(
        detailed_path,
        index=False,
    )

    summary = build_summary(
        dataframe
    )

    summary_path = (
        RESULTS_DIR
        / "agent_evaluation_summary_v3.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    failures = dataframe[
        ~dataframe[
            "final_execution_correct"
        ]
    ]

    failures_path = (
        RESULTS_DIR
        / "agent_evaluation_failures_v3.csv"
    )

    failures.to_csv(
        failures_path,
        index=False,
    )

    print_summary(
        summary
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"- {detailed_path}"
    )

    print(
        f"- {summary_path}"
    )

    print(
        f"- {failures_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()