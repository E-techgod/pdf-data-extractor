from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.pdf_data_extractor.agent import (
    GENERIC_EXTRACTION_HINT,
    INVOICE_EXTRACTION_HINT,
    RECEIPT_EXTRACTION_HINT,
    REPORT_EXTRACTION_HINT,
    RESUME_EXTRACTION_HINT,
    UNREGISTERED_EXTRACTOR_WARNING,
    classify_pdf_with_groq,
    classify_with_groq,
)
from src.pdf_data_extractor.schemas import (
    DocumentExtractionResult,
    GenericDocumentData,
    InvoiceData,
    ReceiptData,
    ReportData,
    ResumeData,
)
from src.pdf_data_extractor.tool_schemas import (
    CLASSIFY_DOCUMENT_TOOL,
)


def make_tool_call(
    *,
    name: str,
    arguments: str | dict,
    call_id: str = "call_123",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_response(
    *,
    tool_calls: list[SimpleNamespace] | None = None,
    content: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def make_fake_client(
    *responses: SimpleNamespace,
) -> Mock:
    client = Mock()
    client.chat.completions.create.side_effect = list(
        responses
    )
    return client


def test_uses_required_classifier_only_on_first_call() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "invoice", "reason": "Contains invoice fields."}'
        ),
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"invoice_number": "123", "total": 50}',
        call_id="call_extract",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
            total=50,
        ),
    )

    first_call = client.chat.completions.create.call_args_list[0]
    assert first_call.kwargs["tools"] == [
        CLASSIFY_DOCUMENT_TOOL
    ]
    assert first_call.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "classify_document"},
    }


def test_routes_to_matching_specialized_extractor_only() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "resume", "reason": "Contains skills and experience."}'
        ),
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_resume_fields",
        arguments=(
            '{"full_name": "Elias Arellano Campos", "skills": ["Python"]}'
        ),
        call_id="call_extract",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Professional Experience. Skills. Education.",
        client=client,
    )

    assert result == DocumentExtractionResult(
        document_type="resume",
        data=ResumeData(
            full_name="Elias Arellano Campos",
            skills=["Python"],
        ),
    )

    second_call = client.chat.completions.create.call_args_list[1]
    assert len(second_call.kwargs["tools"]) == 1
    assert (
        second_call.kwargs["tools"][0]["function"]["name"]
        == "extract_resume_fields"
    )
    assert second_call.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "extract_resume_fields"},
    }
    assert RESUME_EXTRACTION_HINT in second_call.kwargs["messages"][0]["content"]


def test_supported_report_route_returns_validated_data() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "report", "reason": "Contains findings and recommendations."}'
        ),
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_report_fields",
        arguments=(
            '{"title": "Customer Retention Analysis", '
            '"author": "Elias Arellano Campos", '
            '"organization": "American Smart Business LLC", '
            '"report_date": "August 1, 2026", '
            '"executive_summary": "Retention improved during the quarter.", '
            '"methodology": "We analyzed quarterly transaction and support data.", '
            '"findings": ["Customer retention improved by 12%."], '
            '"recommendations": ["Continue monitoring retention monthly."], '
            '"conclusion": "The strategy is working."}'
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Executive Summary. Findings. Recommendations. Conclusion.",
        client=client,
    )

    assert result.document_type == "report"
    assert isinstance(result.data, ReportData)
    assert result.data.title == "Customer Retention Analysis"
    assert result.data.author == "Elias Arellano Campos"
    assert result.data.findings == [
        "Customer retention improved by 12%."
    ]
    assert "document_type" not in result.data.model_dump()
    assert client.chat.completions.create.call_count == 2

    second_call = client.chat.completions.create.call_args_list[1]
    assert len(second_call.kwargs["tools"]) == 1
    assert (
        second_call.kwargs["tools"][0]["function"]["name"]
        == "extract_report_fields"
    )
    assert second_call.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "extract_report_fields"},
    }
    assert REPORT_EXTRACTION_HINT in second_call.kwargs["messages"][0]["content"]


def test_supported_generic_route_returns_validated_data() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "generic", "reason": "Does not match a supported specialized type."}'
        ),
    )
    extract_call = make_tool_call(
        name="extract_generic_fields",
        arguments=(
            '{"title": "Project Kickoff Notes", '
            '"document_date": "August 3, 2026", '
            '"author": "Elias Arellano Campos", '
            '"organization": "Austin Cohort", '
            '"summary": "Notes covering the initial kickoff discussion.", '
            '"key_points": ["Team introductions completed.", "Project timeline was reviewed."], '
            '"document_text_excerpt": "Kickoff meeting notes for the PDF extractor project."}'
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Personal notes for August 3, 2026.",
        client=client,
    )

    assert result.document_type == "generic"
    assert isinstance(result.data, GenericDocumentData)
    assert result.data.title == "Project Kickoff Notes"
    assert result.data.document_date == "August 3, 2026"
    assert result.data.author == "Elias Arellano Campos"
    assert result.data.organization == "Austin Cohort"
    assert "document_type" not in result.data.model_dump()
    assert client.chat.completions.create.call_count == 2

    second_call = client.chat.completions.create.call_args_list[1]
    assert len(second_call.kwargs["tools"]) == 1
    assert (
        second_call.kwargs["tools"][0]["function"]["name"]
        == "extract_generic_fields"
    )
    assert second_call.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "extract_generic_fields"},
    }
    assert GENERIC_EXTRACTION_HINT in second_call.kwargs["messages"][0]["content"]


def test_supported_receipt_route_returns_validated_data() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "receipt", "reason": "Contains receipt number, merchant, totals, and payment method."}'
        ),
    )
    extract_call = make_tool_call(
        name="extract_receipt_fields",
        arguments=(
            '{"merchant": "RIVERSTONE MARKET", '
            '"receipt_number": "RCT-784219", '
            '"transaction_date": "August 3, 2026", '
            '"transaction_time": "12:42 PM", '
            '"items": [{"name": "Whole Milk", "quantity": 2, "amount": 8.58}], '
            '"subtotal": 38.81, '
            '"tax": 3.2, '
            '"total": 42.01, '
            '"payment_method": "Visa Card **** 4412", '
            '"currency": "USD"}'
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Receipt Number: RCT-784219. Merchant: RIVERSTONE MARKET. Total: $42.01.",
        client=client,
    )

    assert result.document_type == "receipt"
    assert isinstance(result.data, ReceiptData)
    assert result.data.merchant == "RIVERSTONE MARKET"
    assert result.data.receipt_number == "RCT-784219"
    assert result.data.total == 42.01
    assert "document_type" not in result.data.model_dump()

    second_call = client.chat.completions.create.call_args_list[1]
    assert len(second_call.kwargs["tools"]) == 1
    assert (
        second_call.kwargs["tools"][0]["function"]["name"]
        == "extract_receipt_fields"
    )
    assert second_call.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "extract_receipt_fields"},
    }
    assert RECEIPT_EXTRACTION_HINT in second_call.kwargs["messages"][0]["content"]


def test_ignores_assistant_prose_and_uses_validated_classification_tool_result() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "invoice", "reason": "Contains invoice number."}'
        ),
    )
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"invoice_number": "123"}',
    )

    client = make_fake_client(
        make_response(
            tool_calls=[classify_call],
            content="This looks like a resume.",
        ),
        make_response(
            tool_calls=[extract_call],
            content="I extracted the invoice fields.",
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result.document_type == "invoice"
    assert isinstance(result.data, InvoiceData)


def test_invoice_route_includes_strict_invoice_field_guidance() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "invoice", "reason": "Contains bill to and total."}'
        ),
    )
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"vendor": "Example Services LLC", "customer": "Acme Corp"}',
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    classify_with_groq(
        "Invoice Number: 123. Bill To: Acme Corp. Vendor: Example Services LLC.",
        client=client,
    )

    second_call = client.chat.completions.create.call_args_list[1]
    assert INVOICE_EXTRACTION_HINT in second_call.kwargs["messages"][0]["content"]


def test_rejects_empty_document() -> None:
    client = Mock()

    with pytest.raises(
        ValueError,
        match="document_text cannot be empty",
    ):
        classify_with_groq(
            "   ",
            client=client,
        )

    client.chat.completions.create.assert_not_called()


def test_raises_when_classifier_is_not_called() -> None:
    client = make_fake_client(make_response())

    with pytest.raises(
        RuntimeError,
        match="Groq did not call required tool: classify_document",
    ):
        classify_with_groq(
            "Some document text",
            client=client,
        )


def test_raises_when_extractor_is_not_called() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "invoice", "reason": "Contains invoice fields."}'
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(),
    )

    with pytest.raises(
        RuntimeError,
        match="Groq did not call required tool: extract_invoice_fields",
    ):
        classify_with_groq(
            "Invoice Number: 123. Amount Due: $50.",
            client=client,
        )


def test_rejects_unknown_tool() -> None:
    tool_call = make_tool_call(
        name="delete_everything",
        arguments="{}",
    )
    client = make_fake_client(
        make_response(tool_calls=[tool_call])
    )

    with pytest.raises(
        RuntimeError,
        match="Expected tool classify_document, received delete_everything",
    ):
        classify_with_groq(
            "Test document",
            client=client,
        )


def test_rejects_invalid_json_arguments() -> None:
    tool_call = make_tool_call(
        name="classify_document",
        arguments="{invalid JSON}",
    )
    client = make_fake_client(
        make_response(tool_calls=[tool_call])
    )

    with pytest.raises(
        ValueError,
        match="Invalid tool arguments",
    ):
        classify_with_groq(
            "Test document",
            client=client,
        )


def test_rejects_non_object_tool_arguments() -> None:
    tool_call = make_tool_call(
        name="classify_document",
        arguments='"not an object"',
    )
    client = make_fake_client(
        make_response(tool_calls=[tool_call])
    )

    with pytest.raises(
        ValueError,
        match="Invalid tool arguments",
    ):
        classify_with_groq(
            "Test document",
            client=client,
        )


def test_rejects_invalid_classification_payload() -> None:
    tool_call = make_tool_call(
        name="classify_document",
        arguments='{"document_type": "memo", "reason": "test"}',
    )
    client = make_fake_client(
        make_response(tool_calls=[tool_call])
    )

    with pytest.raises(ValueError):
        classify_with_groq(
            "Test document",
            client=client,
        )


@patch(
    "src.pdf_data_extractor.agent.extract_pdf_text"
)
def test_classifies_text_extracted_from_pdf(
    mock_extract_pdf_text: Mock,
) -> None:
    mock_extract_pdf_text.return_value = """
    Invoice Number: 987
    Bill To: Example Customer
    Amount Due: $125.00
    """

    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "invoice", "reason": "Contains invoice number and amount due."}'
        ),
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"invoice_number": "987", "total": 125.0}',
        call_id="call_extract",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_pdf_with_groq(
        "data/invoice.pdf",
        client=client,
    )

    assert result == DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="987",
            total=125.0,
        ),
    )
    assert "document_type" not in result.data.model_dump()
    mock_extract_pdf_text.assert_called_once_with(
        "data/invoice.pdf"
    )


def test_supported_resume_route_returns_populated_data_without_nested_document_type() -> None:
    classify_call = make_tool_call(
        name="classify_document",
        arguments=(
            '{"document_type": "resume", "reason": "Contains education and skills."}'
        ),
    )
    extract_call = make_tool_call(
        name="extract_resume_fields",
        arguments='{"full_name": "Elias Arellano Campos", "skills": ["Python", "SQL"]}',
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
    )

    result = classify_with_groq(
        "Professional Experience. Skills. Education.",
        client=client,
    )

    assert result.document_type == "resume"
    assert isinstance(result.data, ResumeData)
    assert result.data.full_name == "Elias Arellano Campos"
    assert result.data.skills == ["Python", "SQL"]
    assert "document_type" not in result.data.model_dump()


def test_build_assistant_tool_message_omits_sdk_only_fields() -> None:
    from src.pdf_data_extractor.agent import (
        _build_assistant_tool_message,
    )

    class FakeFunction:
        name = "classify_document"
        arguments = '{"document_type":"invoice","reason":"sample"}'

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
                    "arguments": (
                        '{"document_type":"invoice","reason":"sample"}'
                    ),
                },
            }
        ],
    }
