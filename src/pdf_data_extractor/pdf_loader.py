from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str | Path) -> str:
    """
    Extract text from every page of a text-based PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        All extracted page text joined into one string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a PDF or no text can be extracted.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"Path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {pdf_path.suffix}"
        )

    reader = PdfReader(pdf_path)

    extracted_pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if not page_text:
            continue

        cleaned_text = page_text.strip()

        if cleaned_text:
            extracted_pages.append(
                f"--- Page {page_number} ---\n{cleaned_text}"
            )

    if not extracted_pages:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The document may be scanned or image-based."
        )

    return "\n\n".join(extracted_pages)