"""
Chaos / adversarial-input tests for src/pdf_data_extractor/pdf_loader.py.

The function's docstring promises exactly two failure modes:
FileNotFoundError and ValueError. These tests probe inputs designed to slip
past that documented contract -- wrong argument types, path-naming edge
cases in pathlib itself, and lower-level library failures that are never
caught or translated.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.pdf_data_extractor.pdf_loader import extract_pdf_text


def test_none_file_path_breaks_documented_exception_contract() -> None:
    # Risk: the docstring promises FileNotFoundError or ValueError for bad
    # input, and the type hint says `str | Path` -- but Python does not
    # enforce type hints at runtime. `Path(None)` inside the function
    # raises a raw TypeError before any of the function's own validation
    # (exists/is_file/suffix checks) ever runs. A caller who defensively
    # wraps this call in `except (FileNotFoundError, ValueError)`, per the
    # documented contract, will NOT catch this and will crash instead.
    with pytest.raises(TypeError):
        extract_pdf_text(None)  # type: ignore[arg-type]


def test_file_literally_named_dot_pdf_is_rejected_as_not_a_pdf(
    tmp_path: Path,
) -> None:
    # Boundary condition: pathlib treats a leading-dot filename with no
    # other dot as an extensionless "hidden" file, not as having a suffix.
    # `Path(".pdf").suffix == ""`, NOT ".pdf". So a file literally named
    # ".pdf" -- whose entire name IS the intended extension -- fails the
    # `pdf_path.suffix.lower() != ".pdf"` check and is rejected with the
    # confusing message "Expected a PDF file, received: " (empty string),
    # even though nothing about the file's content was ever inspected.
    dotfile = tmp_path / ".pdf"
    dotfile.touch()

    with pytest.raises(ValueError, match="Expected a PDF file"):
        extract_pdf_text(dotfile)


def test_page_extract_text_exception_propagates_unwrapped(
    tmp_path: Path,
) -> None:
    # Risk: extract_pdf_text never wraps calls into pypdf. If a page's
    # extract_text() raises (which pypdf does for a range of malformed /
    # corrupted content streams, unsupported filters, etc.), that raw
    # library exception propagates straight through extract_pdf_text
    # completely unmodified -- again breaking the documented
    # FileNotFoundError/ValueError-only contract. A caller written against
    # the docstring has no way to anticipate or cleanly handle this.
    pdf_file = tmp_path / "corrupt.pdf"
    pdf_file.touch()

    broken_page = Mock()
    broken_page.extract_text.side_effect = RuntimeError("corrupt content stream")

    mock_reader = Mock()
    mock_reader.pages = [broken_page]

    with (
        patch(
            "src.pdf_data_extractor.pdf_loader.PdfReader",
            return_value=mock_reader,
        ),
        pytest.raises(RuntimeError, match="corrupt content stream"),
    ):
        extract_pdf_text(pdf_file)


def test_whitespace_only_page_text_is_treated_as_no_text(
    tmp_path: Path,
) -> None:
    # Boundary condition on the truthiness check: `if not page_text:` only
    # filters out falsy values (None, ""). A page that returns a
    # whitespace-only string (e.g. "  \n\t ") is truthy and would pass that
    # guard, but is then stripped via `.strip()` down to an empty string
    # before being appended. This confirms that whitespace-only "text"
    # extracted from a page (common for pages that are technically
    # text-layer-tagged but visually blank) is correctly treated as no
    # usable content rather than silently producing a page entry with
    # empty content.
    pdf_file = tmp_path / "blank.pdf"
    pdf_file.touch()

    blank_page = Mock()
    blank_page.extract_text.return_value = "   \n\t  "

    mock_reader = Mock()
    mock_reader.pages = [blank_page]

    with (
        patch(
            "src.pdf_data_extractor.pdf_loader.PdfReader",
            return_value=mock_reader,
        ),
        pytest.raises(ValueError, match="No text could be extracted"),
    ):
        extract_pdf_text(pdf_file)


def test_uppercase_pdf_extension_with_lowercase_case_sensitive_disk_is_accepted(
    tmp_path: Path,
) -> None:
    # Boundary condition: the suffix check normalizes case
    # (`.suffix.lower() != ".pdf"`), so "REPORT.PDF" is accepted as a valid
    # extension. This locks in that case-insensitive matching is
    # intentional and doesn't regress into rejecting legitimately
    # upper-cased extensions from case-insensitive filesystems or
    # user-supplied paths.
    pdf_file = tmp_path / "REPORT.PDF"
    pdf_file.touch()

    page = Mock()
    page.extract_text.return_value = "Report contents."

    mock_reader = Mock()
    mock_reader.pages = [page]

    with patch(
        "src.pdf_data_extractor.pdf_loader.PdfReader",
        return_value=mock_reader,
    ):
        result = extract_pdf_text(pdf_file)

    assert "Report contents." in result
