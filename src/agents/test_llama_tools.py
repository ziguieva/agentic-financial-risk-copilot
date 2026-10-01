from ollama import chat

from financial_tools import get_current_risk


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "llama3.2:3b"


# ============================================================
# TEST TOOL CALLING
# ============================================================

def main() -> None:

    print("\n========================================")
    print("LLAMA TOOL CALLING TEST")
    print("========================================")

    print(f"Model: {MODEL}")

    response = chat(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial risk assistant. "
                    "Use the provided tool when quantitative "
                    "portfolio risk information is required."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Quel est le risque actuel "
                    "de mon portefeuille ?"
                ),
            },
        ],

        tools=[
            get_current_risk
        ],

        options={
            "temperature": 0.0,
            "num_predict": 200,
        },

        keep_alive="10m",
    )

    print("\n========================================")
    print("CONTENT")
    print("========================================")

    print(
        response.message.content
    )

    print("\n========================================")
    print("TOOL CALLS")
    print("========================================")

    tool_calls = (
        response.message.tool_calls
        or []
    )

    if not tool_calls:

        print(
            "NO TOOL CALL DETECTED"
        )

        return

    for tool_call in tool_calls:

        print(
            f"Tool selected : "
            f"{tool_call.function.name}"
        )

        print(
            f"Arguments     : "
            f"{tool_call.function.arguments}"
        )

    print(
        "\nTool calling test successful."
    )


if __name__ == "__main__":
    main()