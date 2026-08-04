from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.pdf_data_extractor.pdf_loader import extract_pdf_text


def test_rejects_missing_pdf(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(
        FileNotFoundError,
        match="PDF file does not exist",
    ):
        extract_pdf_text(missing_file)


def test_rejects_directory(tmp_path: Path) -> None:
    directory = tmp_path / "documents"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="Path is not a file",
    ):
        extract_pdf_text(directory)


def test_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("Hello", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Expected a PDF file",
    ):
        extract_pdf_text(text_file)


def test_extracts_text_from_all_pages(
    tmp_path: Path,
) -> None:
    pdf_file = tmp_path / "document.pdf"
    pdf_file.touch()

    first_page = Mock()
    first_page.extract_text.return_value = "Invoice Number: 123\nAmount Due: $50"

    second_page = Mock()
    second_page.extract_text.return_value = "Payment Due: August 10, 2026"

    mock_reader = Mock()
    mock_reader.pages = [first_page, second_page]

    with patch(
        "src.pdf_data_extractor.pdf_loader.PdfReader",
        return_value=mock_reader,
    ):
        result = extract_pdf_text(pdf_file)

    assert "--- Page 1 ---" in result
    assert "Invoice Number: 123" in result
    assert "--- Page 2 ---" in result
    assert "Payment Due: August 10, 2026" in result


def test_skips_pages_without_text(
    tmp_path: Path,
) -> None:
    pdf_file = tmp_path / "document.pdf"
    pdf_file.touch()

    empty_page = Mock()
    empty_page.extract_text.return_value = None

    text_page = Mock()
    text_page.extract_text.return_value = "Professional Experience\nEducation\nSkills"

    mock_reader = Mock()
    mock_reader.pages = [empty_page, text_page]

    with patch(
        "src.pdf_data_extractor.pdf_loader.PdfReader",
        return_value=mock_reader,
    ):
        result = extract_pdf_text(pdf_file)

    assert "--- Page 1 ---" not in result
    assert "--- Page 2 ---" in result
    assert "Professional Experience" in result


def test_rejects_pdf_without_extractable_text(
    tmp_path: Path,
) -> None:
    pdf_file = tmp_path / "scanned.pdf"
    pdf_file.touch()

    page = Mock()
    page.extract_text.return_value = None

    mock_reader = Mock()
    mock_reader.pages = [page]

    with (
        patch(
            "src.pdf_data_extractor.pdf_loader.PdfReader",
            return_value=mock_reader,
        ),
        pytest.raises(
            ValueError,
            match="No text could be extracted",
        ),
    ):
        extract_pdf_text(pdf_file)
