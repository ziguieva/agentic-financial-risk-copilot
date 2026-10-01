import json

from financial_tools import compare_volatility_models


def main() -> None:

    print(
        "\n========================================"
    )

    print(
        "RAW MODEL COMPARISON TOOL"
    )

    print(
        "========================================\n"
    )

    result = compare_volatility_models()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()