import ast
import json
from pathlib import Path
from typing import Any

from groq import Groq

from src.pdf_data_extractor.config import get_groq_api_key
from src.pdf_data_extractor.pdf_loader import extract_pdf_text
from src.pdf_data_extractor.schemas import (
    DocumentExtractionResult,
)
from src.pdf_data_extractor.tool_registry import TOOL_REGISTRY
from src.pdf_data_extractor.tool_schemas import TOOLS

DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_TOOL_ROUNDS = 5
EXTRACTION_TOOL_NAMES = {
    "extract_invoice_fields",
    "extract_resume_fields",
    "extract_receipt_fields",
    "extract_report_fields",
    "extract_generic_fields",
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


def _serialize_tool_result(tool_result: Any) -> str:
    if isinstance(tool_result, DocumentExtractionResult):
        return tool_result.model_dump_json()

    return json.dumps(tool_result)


def _parse_final_result(
    final_content: str,
) -> DocumentExtractionResult:
    try:
        parsed_content = json.loads(final_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Groq returned a non-JSON final response."
        ) from exc

    try:
        return DocumentExtractionResult.model_validate(
            parsed_content
        )
    except Exception as exc:
        raise RuntimeError(
            "Groq returned an invalid final structured response."
        ) from exc


def classify_pdf_with_groq(
    file_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> DocumentExtractionResult:
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
) -> DocumentExtractionResult:
    if not document_text.strip():
        raise ValueError("document_text cannot be empty")

    groq_client = client or build_groq_client()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a document extraction assistant.\n\n"
                "Follow this process exactly:\n"
                "1. First call classify_document with an empty JSON object: {}.\n"
                "The document text is already available in the conversation.\n"
                "Do not pass document_text or any other argument to classify_document.\n"
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
                "structured extraction result as JSON with top-level keys "
                'document_type, data, and warnings.\n'
                "12. The data object must match the extracted document type.\n"
                "13. Do not offer additional help or add conversational closing text."
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

    tool_choice = "auto"
    last_extraction_result: (
        DocumentExtractionResult | None
    ) = None

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
            if last_extraction_result is not None:
                return last_extraction_result

            final_content = assistant_message.content

            if not final_content:
                raise RuntimeError(
                    "Groq returned an empty final response."
                )

            return _parse_final_result(final_content)

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

            if (
                tool_name in EXTRACTION_TOOL_NAMES
                and isinstance(
                    tool_result,
                    DocumentExtractionResult,
                )
            ):
                last_extraction_result = tool_result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": _serialize_tool_result(
                        tool_result
                    ),
                }
            )
    raise RuntimeError(
        "Groq exceeded the maximum number of tool rounds."
    )


def analyze_document_with_groq(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> DocumentExtractionResult:
    return classify_with_groq(
        document_text=document_text,
        model=model,
        client=client,
    )
