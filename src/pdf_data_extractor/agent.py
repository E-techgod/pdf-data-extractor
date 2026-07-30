import json
from typing import Any

from groq import Groq

from src.pdf_data_extractor.config import get_groq_api_key
from src.pdf_data_extractor.tool_registry import TOOL_REGISTRY
from src.pdf_data_extractor.tool_schemas import TOOLS
DEFAULT_MODEL =  "llama-3.1-8b-instant" # "openai/gpt-oss-20b"  


def _build_assistant_tool_message(assistant_message: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.content or "",
    }

    if assistant_message.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ]

    return message


def classify_with_groq(document_text: str,*,model: str = DEFAULT_MODEL) -> str:
    if not document_text.strip():
        raise ValueError("document_text cannot be empty")

    client = Groq(api_key=get_groq_api_key())

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a document analysis assistant. "
                "Use the available classification tool to classify "
                "the provided document. After receiving the tool result, "
                "explain the classification briefly."
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

    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
        temperature=0,
    )

    assistant_message = first_response.choices[0].message

    if not assistant_message.tool_calls:
        raise RuntimeError(
            "Groq did not return a tool call."
        )

    messages.append(
        _build_assistant_tool_message(assistant_message)
    )

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name

        if tool_name not in TOOL_REGISTRY:
            raise ValueError(
                f"Unknown tool requested: {tool_name}"
            )

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid tool arguments for {tool_name}"
            ) from exc

        tool_function = TOOL_REGISTRY[tool_name]
        tool_result = tool_function(**arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            }
        )

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )

    final_content = final_response.choices[0].message.content

    if not final_content:
        raise RuntimeError(
            "Groq returned an empty final response."
        )

    return final_content
