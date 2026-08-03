import pytest
from pydantic import ValidationError

from src.pdf_data_extractor.agent import (
    _build_assistant_tool_message,
)
from src.pdf_data_extractor.tools import classify_document

from src.pdf_data_extractor.tools import extract_invoice_fields


def test_classifies_invoice() -> None:
    document = """
    Invoice Number: 32107
    Bill To: Elias Arellano
    Amount Due: $515.00
    Payment Due: August 5, 2026
    """

    result = classify_document(document)

    assert result["document_type"] == "invoice"
    assert "keyword match" in result["reason"]


def test_classifies_resume() -> None:
    document = """
    Elias Arellano Campos

    Professional Experience
    Software Developer

    Education
    Bachelor of Science in Computer Science

    Skills
    Python, SQL, Machine Learning
    """

    result = classify_document(document)

    assert result["document_type"] == "resume"


def test_classifies_receipt() -> None:
    document = """
    Store Receipt
    Subtotal: $45.00
    Sales Tax: $3.71
    Payment Method: Visa
    """

    result = classify_document(document)

    assert result["document_type"] == "receipt"


def test_classifies_report() -> None:
    document = """
    Executive Summary

    Methodology
    The analysis was conducted using customer transaction data.

    Findings
    Customer retention improved by 12%.

    Recommendations
    Continue monitoring retention monthly.
    """

    result = classify_document(document)

    assert result["document_type"] == "report"


def test_classifies_unknown_document_as_generic() -> None:
    document = """
    This is a short personal note about tomorrow's meeting.
    """

    result = classify_document(document)

    assert result["document_type"] == "generic"


def test_rejects_empty_document() -> None:
    with pytest.raises(
        ValueError,
        match="document_text cannot be empty",
    ):
        classify_document("   ")


def test_build_assistant_tool_message_omits_sdk_only_fields() -> None:
    class FakeFunction:
        name = "classify_document"
        arguments = '{"document_text":"sample"}'

    class FakeToolCall:
        id = "call_123"
        type = "function"
        function = FakeFunction()

    class FakeAssistantMessage:
        content = None
        tool_calls = [FakeToolCall()]
        annotations = [{"unsupported": True}]

    result = _build_assistant_tool_message(
        FakeAssistantMessage()
    )

    assert result == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "classify_document",
                    "arguments": '{"document_text":"sample"}',
                },
            }
        ],
    }

def test_extracts_invoice_fields() -> None:
    result = extract_invoice_fields(
        invoice_number="INV-1001",
        vendor="Example Services LLC",
        customer="Elias Arellano",
        invoice_date="August 1, 2026",
        due_date="August 15, 2026",
        subtotal=500.00,
        tax=41.25,
        total=541.25,
        currency="usd",
    )

    assert result["document_type"] == "invoice"

    invoice = result["data"]

    assert invoice["invoice_number"] == "INV-1001"
    assert invoice["vendor"] == "Example Services LLC"
    assert invoice["total"] == 541.25
    assert invoice["currency"] == "USD"


def test_invoice_allows_missing_fields() -> None:
    result = extract_invoice_fields(
        vendor="Example Services LLC",
        total=125.00,
    )

    invoice = result["data"]

    assert invoice["vendor"] == "Example Services LLC"
    assert invoice["invoice_number"] is None
    assert invoice["total"] == 125.00


def test_invoice_rejects_negative_total() -> None:
    with pytest.raises(ValidationError):
        extract_invoice_fields(
            vendor="Example Services LLC",
            total=-50.00,
        )

from src.pdf_data_extractor.agent import analyze_document_with_groq


def main() -> None:
    document = """
    INVOICE

    High End Locksmiths, LLC
    Invoice Number: 32107
    Invoice Date: July 29, 2026

    Bill To:
    Elias Arellano Campos

    Service:
    Volkswagen Jetta 2021 replacement key

    Subtotal: $475.00
    Tax: $40.00
    Total Due: $515.00

    Payment Due: August 5, 2026
    """

    result = analyze_document_with_groq(document)

    print(result)


if __name__ == "__main__":
    main()