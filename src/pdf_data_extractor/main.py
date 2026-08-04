import sys
from pathlib import Path

from src.pdf_data_extractor.agent import classify_pdf_with_groq


def main() -> None:  # Passed invoice, receipt, resume, report
    pdf_arg = (
        sys.argv[1] if len(sys.argv) > 1 else "data/receipt.pdf"
    )  # resume, receipt, generic, invoice, report
    pdf_path = Path(pdf_arg)
    result = classify_pdf_with_groq(pdf_path)

    print("\nFinal response:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
