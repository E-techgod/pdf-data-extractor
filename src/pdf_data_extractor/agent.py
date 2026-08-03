import ast
import json
import re
from pathlib import Path
from typing import Any

from groq import Groq

from src.pdf_data_extractor.config import get_groq_api_key
from src.pdf_data_extractor.pdf_loader import extract_pdf_text
from src.pdf_data_extractor.schemas import (
    DocumentClassification,
    DocumentExtractionResult,
    EmptyExtractionData,
)
from src.pdf_data_extractor.tool_registry import (
    SPECIALIZED_TOOL_NAMES,
    SPECIALIZED_TOOL_SCHEMAS,
    TOOL_REGISTRY,
)
from src.pdf_data_extractor.tool_schemas import (
    CLASSIFY_DOCUMENT_TOOL,
)

DEFAULT_MODEL = "llama-3.1-8b-instant"
UNREGISTERED_EXTRACTOR_WARNING = (
    "No specialized extractor is registered for document_type "
    "'{document_type}'."
)
RESUME_EXTRACTION_HINT = (
    "For resume extraction, return education and experience as arrays "
    "of plain strings or well-structured objects only. Do not concatenate "
    "multiple semantic fields into one value. Split institution, degree, "
    "field, minor, and graduation date into their matching properties. "
    "Split company, title, dates, location, and responsibilities into "
    "their matching properties. Capture the top contact block into "
    "full_name, email, phone, and location whenever those values appear. "
    "Do not leave those fields null when the resume header explicitly shows "
    "them. When the resume has a labeled skills section, extract those "
    "skills into the skills array. If the skills line is comma-separated, "
    "split it into separate skill entries and preserve the written skill names. "
    "For education, keep degree type separate from field and minor: "
    "for example, use degree='Bachelor of Science', field='Computer Science', "
    "and minor='Mathematics', not merged phrases such as "
    "'Bachelor of Science in Computer Science' or 'Minor in Mathematics'."
)
INVOICE_EXTRACTION_HINT = (
    "For invoice extraction, do not concatenate multiple semantic fields "
    "into one value. Return only the issuing entity in vendor and only the "
    "billed party in customer. Do not include addresses, service provider "
    "labels, departments, or unrelated organization names in vendor or customer."
)
RECEIPT_EXTRACTION_HINT = (
    "For receipt extraction, keep merchant, receipt number, date, time, "
    "payment method, totals, and line items separated by meaning. Do not "
    "merge multiple semantic fields into merchant or payment_method. For "
    "line items, use item name only in name, quantity only in quantity or "
    "qty, and amount only in amount."
)
REPORT_EXTRACTION_HINT = (
    "For report extraction, keep title, author, organization, report date, "
    "executive summary, methodology, findings, recommendations, and conclusion "
    "separated by meaning. Do not merge multiple sections into one field. "
    "Use findings and recommendations only for list-style report points."
)
GENERIC_EXTRACTION_HINT = (
    "For generic extraction, keep title, date, author, organization, summary, "
    "key points, and excerpt separated by meaning. Do not merge multiple "
    "semantic fields into title, author, or organization."
)
CLASSIFICATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "invoice": (
        "invoice",
        "invoice number",
        "amount due",
        "bill to",
        "due date",
        "subtotal",
        "tax",
        "balance due",
    ),
    "receipt": (
        "receipt",
        "transaction",
        "payment method",
        "change due",
        "cash",
        "card",
        "qty",
        "merchant",
    ),
    "report": (
        "report",
        "executive summary",
        "methodology",
        "findings",
        "recommendations",
        "conclusion",
        "analysis",
    ),
    "resume": (
        "resume",
        "curriculum vitae",
        "professional summary",
        "work experience",
        "professional experience",
        "education",
        "skills",
        "certifications",
        "projects",
        "linkedin",
    ),
}


def _count_keyword_occurrences(
    normalized_text: str,
    keyword: str,
) -> int:
    if " " in keyword:
        return normalized_text.count(keyword)

    return len(
        re.findall(
            rf"\b{re.escape(keyword)}\b",
            normalized_text,
        )
    )


def _score_document_types(
    document_text: str,
) -> dict[str, int]:
    normalized_text = document_text.lower()

    return {
        document_type: sum(
            _count_keyword_occurrences(
                normalized_text,
                keyword,
            )
            for keyword in keywords
        )
        for document_type, keywords in CLASSIFICATION_KEYWORDS.items()
    }


def _classify_document_by_keywords(
    document_text: str,
) -> DocumentClassification:
    scores = _score_document_types(document_text)
    best_document_type = max(
        scores,
        key=scores.get,
    )
    best_score = scores[best_document_type]
    competing_scores = sorted(
        scores.values(),
        reverse=True,
    )
    second_best_score = (
        competing_scores[1]
        if len(competing_scores) > 1
        else 0
    )

    if best_score < 2 or best_score == second_best_score:
        return DocumentClassification(
            document_type="generic",
            reason=(
                "Keyword signals do not strongly match invoice, receipt, "
                "report, or resume patterns."
            ),
        )

    return DocumentClassification(
        document_type=best_document_type,
        reason=(
            f"Keyword scoring matched {best_document_type} patterns "
            f"more strongly than other supported document types."
        ),
    )

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


def _build_required_tool_choice(
    tool_name: str,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool_name},
    }


def _build_assistant_tool_message(
    assistant_message: Any,
) -> dict[str, Any]:
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


def _request_required_tool_call(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tool_schema: dict[str, Any],
    expected_tool_name: str,
) -> Any:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[tool_schema],
        tool_choice=_build_required_tool_choice(
            expected_tool_name
        ),
        temperature=0,
    )

    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:
        raise RuntimeError(
            f"Groq did not call required tool: {expected_tool_name}"
        )

    if len(tool_calls) != 1:
        raise RuntimeError(
            f"Expected one tool call, received {len(tool_calls)}"
        )

    tool_call = tool_calls[0]

    if tool_call.function.name != expected_tool_name:
        raise RuntimeError(
            f"Expected tool {expected_tool_name}, "
            f"received {tool_call.function.name}"
        )

    return tool_call


def _execute_tool_call(
    tool_call: Any,
) -> Any:
    tool_name = tool_call.function.name

    tool_function = TOOL_REGISTRY.get(tool_name)
    if tool_function is None:
        raise ValueError(
            f"Unknown tool requested: {tool_name}"
        )

    arguments = _parse_tool_arguments(
        tool_call.function.arguments,
        tool_name=tool_name,
    )

    try:
        return tool_function(**arguments)
    except TypeError as exc:
        raise ValueError(
            f"Invalid arguments for tool: {tool_name}"
        ) from exc


def _empty_result_for_document_type(
    document_type: str,
) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_type=document_type,
        data=EmptyExtractionData(),
        warnings=[
            UNREGISTERED_EXTRACTOR_WARNING.format(
                document_type=document_type
            )
        ],
    )


def build_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())


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


def classify_with_groq(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> DocumentExtractionResult:
    if not document_text.strip():
        raise ValueError("document_text cannot be empty")

    groq_client = client or build_groq_client()

    classification_messages = [
        {
            "role": "system",
            "content": (
                "Classify the document into exactly one of these types: "
                "invoice, resume, receipt, report, or generic. "
                "Call classify_document with the chosen document_type "
                "and a brief reason grounded in the document. "
                "Do not provide prose outside the tool call."
            ),
        },
        {
            "role": "user",
            "content": document_text,
        },
    ]

    classification_tool_call = _request_required_tool_call(
        client=groq_client,
        model=model,
        messages=classification_messages,
        tool_schema=CLASSIFY_DOCUMENT_TOOL,
        expected_tool_name="classify_document",
    )

    classification_result = _execute_tool_call(
        classification_tool_call
    )

    if not isinstance(
        classification_result,
        DocumentClassification,
    ):
        raise RuntimeError(
            "classify_document returned an invalid result."
        )

    keyword_classification = (
        _classify_document_by_keywords(document_text)
    )
    classification_result = keyword_classification

    document_type = classification_result.document_type
    specialized_tool_name = SPECIALIZED_TOOL_NAMES.get(
        document_type
    )

    if specialized_tool_name is None:
        return _empty_result_for_document_type(
            document_type
        )

    extraction_messages = [
        {
            "role": "system",
            "content": (
                "Extract only the requested structured fields from the "
                f"{document_type} document. Use only information explicitly "
                "present in the document. Omit unavailable optional fields "
                "or use null. Do not provide prose outside the tool call. "
                + (
                    RESUME_EXTRACTION_HINT
                    if document_type == "resume"
                    else INVOICE_EXTRACTION_HINT
                    if document_type == "invoice"
                    else RECEIPT_EXTRACTION_HINT
                    if document_type == "receipt"
                    else REPORT_EXTRACTION_HINT
                    if document_type == "report"
                    else GENERIC_EXTRACTION_HINT
                    if document_type == "generic"
                    else ""
                )
            ),
        },
        {
            "role": "user",
            "content": (
                f"The document has already been classified as {document_type}. "
                "Use the available extraction tool only.\n\n"
                f"{document_text}"
            ),
        },
    ]

    extraction_tool_call = _request_required_tool_call(
        client=groq_client,
        model=model,
        messages=extraction_messages,
        tool_schema=SPECIALIZED_TOOL_SCHEMAS[
            document_type
        ],
        expected_tool_name=specialized_tool_name,
    )

    extraction_result = _execute_tool_call(
        extraction_tool_call
    )

    if not isinstance(
        extraction_result,
        DocumentExtractionResult,
    ):
        raise RuntimeError(
            f"{specialized_tool_name} returned an invalid result."
        )

    return extraction_result


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
