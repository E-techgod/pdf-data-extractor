from src.pdf_data_extractor.agent import classify_with_groq


def main() -> None:
    document = """
    Invoice Number: 32107
    Bill To: Elias Arellano Campos
    Service: Volkswagen Jetta 2021 replacement key
    Amount Due: $515.00
    Payment Due: August 5, 2026
    """

    result = classify_with_groq(document)

    print("\nFinal response:")
    print(result)


if __name__ == "__main__":
    main()