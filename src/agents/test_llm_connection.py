import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gpt-5.6-luna"


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()


# ============================================================
# CHECK API KEY
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found.\n"
        "Check that your .env file contains:\n"
        "OPENAI_API_KEY=sk-..."
    )


# ============================================================
# CREATE CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key
)


# ============================================================
# TEST CONNECTION
# ============================================================

def test_connection() -> None:

    print("\n========================================")
    print("OPENAI CONNECTION TEST")
    print("========================================")

    print(f"Model: {MODEL}")
    print("Sending request...")

    response = client.responses.create(
        model=MODEL,
        input=(
            "Reply with exactly this sentence: "
            "Financial Risk Agent connection successful."
        ),
    )

    print("\n========================================")
    print("LLM RESPONSE")
    print("========================================")

    print(
        response.output_text
    )

    print("\n========================================")
    print("SUCCESS")
    print("========================================")

    print(
        "OpenAI API connection validated successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_connection()