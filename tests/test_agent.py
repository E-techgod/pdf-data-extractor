from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.pdf_data_extractor.agent import (
    MAX_TOOL_ROUNDS,
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


def make_tool_call(
    *,
    name: str = "classify_document",
    arguments: str | dict,
    call_id: str = "call_123",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
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
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )

    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def make_fake_client(
    *responses: SimpleNamespace,
) -> Mock:
    client = Mock()
    client.chat.completions.create.side_effect = list(
        responses
    )
    return client


def test_executes_tool_and_returns_typed_final_response() -> None:
    tool_call = make_tool_call(arguments="{}")
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
            total=50,
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == final_result
    assert client.chat.completions.create.call_count == 2


def test_sends_tool_result_back_to_model() -> None:
    tool_call = make_tool_call(arguments="{}")
    final_result = DocumentExtractionResult(
        document_type="resume",
        data=ResumeData(
            full_name="Elias Arellano Campos",
        ),
        warnings=["Not used in assertion."],
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    classify_with_groq(
        "Professional Experience. Education. Skills.",
        client=client,
    )

    second_call = (
        client.chat.completions.create.call_args_list[1]
    )
    sent_messages = second_call.kwargs["messages"]

    tool_messages = [
        message
        for message in sent_messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_123"
    assert tool_messages[0]["name"] == "classify_document"
    assert (
        '"document_type": "resume"'
        in tool_messages[0]["content"]
    )


def test_uses_auto_tool_choice_on_first_call() -> None:
    tool_call = make_tool_call(arguments="{}")
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    first_call = client.chat.completions.create.call_args_list[0]

    assert first_call.kwargs["tool_choice"] == "auto"


def test_uses_auto_tool_choice_after_first_call() -> None:
    tool_call = make_tool_call(arguments="{}")
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    second_call = client.chat.completions.create.call_args_list[1]

    assert second_call.kwargs["tool_choice"] == "auto"


def test_accepts_python_dict_style_tool_arguments() -> None:
    tool_call = make_tool_call(arguments="{}")
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
            total=50,
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == final_result


def test_accepts_preparsed_tool_arguments() -> None:
    tool_call = make_tool_call(arguments={})
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    result = classify_with_groq(
        "Professional Experience. Education. Skills.",
        client=client,
    )

    assert result == final_result


def test_ignores_model_supplied_document_text() -> None:
    tool_call = make_tool_call(
        arguments=(
            '{"document_text": "Receipt. Subtotal. Sales tax."}'
        )
    )
    final_result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="123",
        ),
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content=final_result.model_dump_json()
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == final_result


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


def test_raises_when_model_returns_no_tool_call() -> None:
    client = Mock()
    client.chat.completions.create.return_value = (
        make_response()
    )

    with pytest.raises(
        RuntimeError,
        match="Groq did not return a tool call",
    ):
        classify_with_groq(
            "Some document text",
            client=client,
        )


def test_rejects_unknown_tool() -> None:
    tool_call = make_tool_call(
        name="delete_everything",
        arguments='{"document_text": "Test"}',
    )

    client = Mock()
    client.chat.completions.create.return_value = (
        make_response(tool_calls=[tool_call])
    )

    with pytest.raises(
        ValueError,
        match="Unknown tool requested",
    ):
        classify_with_groq(
            "Test",
            client=client,
        )


def test_rejects_invalid_json_arguments() -> None:
    tool_call = make_tool_call(
        arguments="{invalid JSON}",
    )

    client = Mock()
    client.chat.completions.create.return_value = (
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
        arguments='"not an object"',
    )

    client = Mock()
    client.chat.completions.create.return_value = (
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


def test_rejects_empty_final_response() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(content=None),
    )

    with pytest.raises(
        RuntimeError,
        match="Groq returned an empty final response",
    ):
        classify_with_groq(
            "Invoice Number: 123. Amount Due: $50.",
            client=client,
        )


def test_rejects_non_json_final_response() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(content="The document is an invoice."),
    )

    with pytest.raises(
        RuntimeError,
        match="non-JSON final response",
    ):
        classify_with_groq(
            "Invoice Number: 123. Amount Due: $50.",
            client=client,
        )


def test_supports_multiple_tool_rounds() -> None:
    classify_call = make_tool_call(
        arguments="{}",
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments=(
            '{"invoice_number": "123", '
            '"vendor": "ACME Corp", '
            '"total": 50, '
            '"currency": "usd"}'
        ),
        call_id="call_extract",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Bill To: Client. Amount Due: $50.",
        client=client,
    )

    assert result.document_type == "invoice"
    assert isinstance(result.data, InvoiceData)
    assert result.data.invoice_number == "123"
    assert client.chat.completions.create.call_count == 3

    third_call = client.chat.completions.create.call_args_list[2]
    tool_messages = [
        message
        for message in third_call.kwargs["messages"]
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(tool_messages) == 2
    assert tool_messages[0]["name"] == "classify_document"
    assert tool_messages[1]["name"] == "extract_invoice_fields"
    assert (
        '"invoice_number":"123"'
        in tool_messages[1]["content"]
    )


def test_supports_receipt_extraction_tool_round() -> None:
    classify_call = make_tool_call(
        arguments="{}",
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_receipt_fields",
        arguments=(
            '{"merchant": "HEB", '
            '"receipt_number": "1001", '
            '"items": ["Milk"], '
            '"subtotal": 12.5, '
            '"tax": 1.03, '
            '"total": 13.53, '
            '"payment_method": "visa", '
            '"currency": "usd"}'
        ),
        call_id="call_extract_receipt",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_with_groq(
        "Store Receipt. Subtotal: $12.50. Sales Tax: $1.03. "
        "Total: $13.53. Payment Method: Visa.",
        client=client,
    )

    assert result.document_type == "receipt"
    assert isinstance(result.data, ReceiptData)
    assert result.data.receipt_number == "1001"

    third_call = client.chat.completions.create.call_args_list[2]
    tool_messages = [
        message
        for message in third_call.kwargs["messages"]
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(tool_messages) == 2
    assert tool_messages[1]["name"] == "extract_receipt_fields"
    assert '"document_type":"receipt"' in tool_messages[1]["content"]


def test_supports_report_extraction_tool_round() -> None:
    classify_call = make_tool_call(
        arguments="{}",
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_report_fields",
        arguments=(
            '{"title": "Customer Retention Analysis", '
            '"author": "Elias Arellano Campos", '
            '"findings": ["Retention improved by 12%."], '
            '"recommendations": ["Continue monitoring monthly."], '
            '"conclusion": "The strategy is working."}'
        ),
        call_id="call_extract_report",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_with_groq(
        "Executive Summary. Methodology. Findings. Recommendations. Conclusion.",
        client=client,
    )

    assert result.document_type == "report"
    assert isinstance(result.data, ReportData)
    assert result.data.title == "Customer Retention Analysis"


def test_supports_generic_extraction_tool_round() -> None:
    classify_call = make_tool_call(
        arguments="{}",
        call_id="call_classify",
    )
    extract_call = make_tool_call(
        name="extract_generic_fields",
        arguments=(
            '{"title": "Project Kickoff Notes", '
            '"document_date": "August 2, 2026", '
            '"key_points": ["Reviewed next steps."], '
            '"summary": "Notes from the kickoff meeting."}'
        ),
        call_id="call_extract_generic",
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_with_groq(
        "This is a short personal note about the kickoff meeting tomorrow.",
        client=client,
    )

    assert result.document_type == "generic"
    assert isinstance(result.data, GenericDocumentData)
    assert result.data.title == "Project Kickoff Notes"


def test_supports_extraction_tool_on_first_round() -> None:
    extract_call = make_tool_call(
        name="extract_generic_fields",
        arguments=(
            '{"title": "AI Engineer Roadmap", '
            '"summary": "A six-month roadmap.", '
            '"key_points": ["Build weekly."]}'
        ),
        call_id="call_extract_generic",
    )

    client = make_fake_client(
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_with_groq(
        "AI Engineer Roadmap: Zero to Hired in 6 Months.",
        client=client,
    )

    assert result.document_type == "generic"
    assert isinstance(result.data, GenericDocumentData)
    assert result.data.title == "AI Engineer Roadmap"

    second_call = client.chat.completions.create.call_args_list[1]
    tool_messages = [
        message
        for message in second_call.kwargs["messages"]
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "extract_generic_fields"
    assert '"document_type":"generic"' in tool_messages[0]["content"]


def test_raises_when_tool_round_limit_is_exceeded() -> None:
    tool_call = make_tool_call(arguments="{}")
    responses = [
        make_response(tool_calls=[tool_call])
        for _ in range(MAX_TOOL_ROUNDS + 1)
    ]
    client = make_fake_client(*responses)

    with pytest.raises(
        RuntimeError,
        match="maximum number of tool rounds",
    ):
        classify_with_groq(
            "Invoice Number: 123. Amount Due: $50.",
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

    classify_call = make_tool_call(arguments="{}")
    extract_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"invoice_number": "987", "total": 125.0}',
    )

    client = make_fake_client(
        make_response(tool_calls=[classify_call]),
        make_response(tool_calls=[extract_call]),
        make_response(content="ignored"),
    )

    result = classify_pdf_with_groq(
        "data/invoice.pdf",
        client=client,
    )

    assert result.document_type == "invoice"
    assert isinstance(result.data, InvoiceData)
    assert result.data.invoice_number == "987"
    mock_extract_pdf_text.assert_called_once_with(
        "data/invoice.pdf"
    )
