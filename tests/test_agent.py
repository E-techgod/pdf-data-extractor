from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.pdf_data_extractor.agent import (
    CLASSIFY_DOCUMENT_TOOL_CHOICE,
    MAX_TOOL_ROUNDS,
    classify_pdf_with_groq,
    classify_with_groq,
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
        choices=[
            SimpleNamespace(message=message)
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


def test_executes_tool_and_returns_final_response() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is an invoice."
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == "The document is an invoice."
    assert client.chat.completions.create.call_count == 2


def test_sends_tool_result_back_to_model() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is a resume."
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


def test_forces_expected_tool_on_first_call() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is an invoice."
        ),
    )

    classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    first_call = client.chat.completions.create.call_args_list[0]

    assert (
        first_call.kwargs["tool_choice"]
        == CLASSIFY_DOCUMENT_TOOL_CHOICE
    )


def test_uses_auto_tool_choice_after_first_call() -> None:
    tool_call = make_tool_call(arguments="{}")

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is an invoice."
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

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is an invoice."
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == "The document is an invoice."


def test_accepts_preparsed_tool_arguments() -> None:
    tool_call = make_tool_call(arguments={})

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is a resume."
        ),
    )

    result = classify_with_groq(
        "Professional Experience. Education. Skills.",
        client=client,
    )

    assert result == "The document is a resume."


def test_ignores_model_supplied_document_text() -> None:
    tool_call = make_tool_call(
        arguments=(
            '{"document_text": "Receipt. Subtotal. Sales tax."}'
        )
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The document is an invoice."
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == "The document is an invoice."


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
        make_response(
            content=(
                "The document is an invoice. "
                "Invoice 123 from ACME Corp totals $50."
            )
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Bill To: Client. Amount Due: $50.",
        client=client,
    )

    assert "Invoice 123" in result
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
        '"invoice_number": "123"'
        in tool_messages[1]["content"]
    )


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

    tool_call = make_tool_call(
        arguments=(
            '{"document_text": '
            '"Invoice Number: 987. '
            'Bill To: Example Customer. '
            'Amount Due: $125.00."}'
        )
    )

    client = make_fake_client(
        make_response(tool_calls=[tool_call]),
        make_response(
            content="The PDF is classified as an invoice."
        ),
    )

    result = classify_pdf_with_groq(
        "data/invoice.pdf",
        client=client,
    )

    assert result == (
        "The PDF is classified as an invoice."
    )
    mock_extract_pdf_text.assert_called_once_with(
        "data/invoice.pdf"
    )
