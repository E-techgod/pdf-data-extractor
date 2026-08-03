from pathlib import Path

from src.pdf_data_extractor.agent import classify_pdf_with_groq


def main() -> None:
    pdf_path = Path("data/sample.pdf")

    result = classify_pdf_with_groq(pdf_path)

    print("\nFinal response:")
    print(result)


if __name__ == "__main__":
    main()