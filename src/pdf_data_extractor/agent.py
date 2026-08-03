import ast
import json
from pathlib import Path
from typing import Any

from groq import Groq

from src.pdf_data_extractor.config import get_groq_api_key
from src.pdf_data_extractor.tool_registry import TOOL_REGISTRY
from src.pdf_data_extractor.tool_schemas import TOOLS
from src.pdf_data_extractor.pdf_loader import extract_pdf_text

DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_TOOL_ROUNDS = 5
CLASSIFY_DOCUMENT_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "classify_document"},
}


def _build_assistant_tool_message(assistant_message: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.content or "",
    }

    if assistant_message.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ]

    return message


def _parse_tool_arguments(
    raw_arguments: Any,
    *,
    tool_name: str,
) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        raise ValueError(
            f"Invalid tool arguments for {tool_name}"
        )

    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        try:
            parsed_arguments = ast.literal_eval(
                raw_arguments
            )
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid tool arguments for {tool_name}"
            ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            f"Invalid tool arguments for {tool_name}"
        )

    return parsed_arguments


def classify_pdf_with_groq(
    file_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> str:
    document_text = extract_pdf_text(file_path)

    return classify_with_groq(
        document_text=document_text,
        model=model,
        client=client,
    )


def build_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())


def classify_with_groq(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> str:
    if not document_text.strip():
        raise ValueError("document_text cannot be empty")

    groq_client = client or build_groq_client()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a document extraction assistant.\n\n"
                "Follow this process exactly:\n"
                "1. First call classify_document using the complete document text.\n"
                "2. Read the classification tool result.\n"
                "3. If the result is invoice, call extract_invoice_fields.\n"
                "4. If the result is resume, call extract_resume_fields.\n"
                "5. If the result is receipt, call extract_receipt_fields.\n"
                "6. If the result is report, call extract_report_fields.\n"
                "7. If the result is generic, call extract_generic_fields.\n"
                "8. Use only information explicitly found in the document.\n"
                "9. Never infer or invent missing values.\n"
                "10. Omit unavailable optional arguments or use null.\n"
                "11. After all required tools are complete, return the final "
                "structured extraction result.\n"
                "12. Do not offer additional help or add conversational closing text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify the following document:\n\n"
                f"{document_text}"
            ),
        },
    ]

    tool_choice: str | dict[str, Any] = (
        CLASSIFY_DOCUMENT_TOOL_CHOICE
    )

    for round_index in range(MAX_TOOL_ROUNDS + 1):
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            temperature=0,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if round_index == 0 and not tool_calls:
            raise RuntimeError(
                "Groq did not return a tool call."
            )

        if not tool_calls:
            final_content = assistant_message.content

            if not final_content:
                raise RuntimeError(
                    "Groq returned an empty final response."
                )

            return final_content

        if round_index == MAX_TOOL_ROUNDS:
            raise RuntimeError(
                "Groq exceeded the maximum number of tool rounds."
            )

        messages.append(
            _build_assistant_tool_message(
                assistant_message
            )
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            if tool_name not in TOOL_REGISTRY:
                raise ValueError(
                    f"Unknown tool requested: {tool_name}"
                )

            arguments = _parse_tool_arguments(
                tool_call.function.arguments,
                tool_name=tool_name,
            )

            tool_function = TOOL_REGISTRY[tool_name]

            if tool_name == "classify_document":
                arguments = {
                    "document_text": document_text
                }

            try:
                tool_result = tool_function(**arguments)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid arguments for tool: {tool_name}"
                ) from exc

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                }
            )

        tool_choice = "auto"

    raise RuntimeError(
        "Groq exceeded the maximum number of tool rounds."
    )


def analyze_document_with_groq(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> str:
    return classify_with_groq(
        document_text=document_text,
        model=model,
        client=client,
    )
