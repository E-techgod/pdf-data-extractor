from pathlib import Path

from src.pdf_data_extractor.agent import analyze_document_with_groq


def main() -> None:
    pdf_path = Path("data/invoice.pdf")

    with open(pdf_path, "r") as f:
        document = f.read()

    result = analyze_document_with_groq(document)

    print("\nFinal response:")
    print(result)


if __name__ == "__main__":
    main()

