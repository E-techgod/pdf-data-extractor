import pytest

from src.pdf_data_extractor.agent import (
    _build_assistant_tool_message,
)
from src.pdf_data_extractor.tools import classify_document


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
