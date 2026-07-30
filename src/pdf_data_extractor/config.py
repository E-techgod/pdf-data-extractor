import os

from dotenv import load_dotenv

load_dotenv()

def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return api_key