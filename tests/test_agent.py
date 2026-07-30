from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.pdf_data_extractor.agent import classify_with_groq


def make_tool_call(
    *,
    name: str = "classify_document",
    arguments: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_first_response(
    tool_call: SimpleNamespace | None,
) -> SimpleNamespace:
    message = Mock()
    message.tool_calls = (
        [tool_call]
        if tool_call is not None
        else None
    )
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
    }

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def make_final_response(
    content: str | None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message)
        ]
    )


def make_fake_client(
    first_response: SimpleNamespace,
    final_response: SimpleNamespace,
) -> Mock:
    client = Mock()

    client.chat.completions.create.side_effect = [
        first_response,
        final_response,
    ]

    return client

def test_executes_tool_and_returns_final_response() -> None:
    tool_call = make_tool_call(
        arguments=(
            '{"document_text": '
            '"Invoice Number: 123. Amount Due: $50."}'
        )
    )

    client = make_fake_client(
        first_response=make_first_response(tool_call),
        final_response=make_final_response(
            "The document is an invoice."
        ),
    )

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result == "The document is an invoice."
    assert client.chat.completions.create.call_count == 2

def test_sends_tool_result_back_to_model() -> None:
    tool_call = make_tool_call(
        arguments=(
            '{"document_text": '
            '"Professional Experience. Education. Skills."}'
        )
    )

    client = make_fake_client(
        first_response=make_first_response(tool_call),
        final_response=make_final_response(
            "The document is a resume."
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

    tool_message = tool_messages[0]

    assert tool_message["tool_call_id"] == "call_123"
    assert tool_message["name"] == "classify_document"
    assert '"document_type": "resume"' in tool_message["content"]

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
        make_first_response(None)
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
        make_first_response(tool_call)
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
        make_first_response(tool_call)
    )

    with pytest.raises(
        ValueError,
        match="Invalid tool arguments",
    ):
        classify_with_groq(
            "Test document",
            client=client,
        )


def test_rejects_missing_required_tool_argument() -> None:
    tool_call = make_tool_call(
        arguments="{}",
    )

    client = Mock()
    client.chat.completions.create.return_value = (
        make_first_response(tool_call)
    )

    with pytest.raises(
        ValueError,
        match="Invalid arguments for tool",
    ):
        classify_with_groq(
            "Test document",
            client=client,
        )


def test_rejects_empty_final_response() -> None:
    tool_call = make_tool_call(
        arguments=(
            '{"document_text": '
            '"Invoice Number: 123. Amount Due: $50."}'
        )
    )

    client = make_fake_client(
        first_response=make_first_response(tool_call),
        final_response=make_final_response(None),
    )

    with pytest.raises(
        RuntimeError,
        match="Groq returned an empty final response",
    ):
        classify_with_groq(
            "Invoice Number: 123. Amount Due: $50.",
            client=client,
        )