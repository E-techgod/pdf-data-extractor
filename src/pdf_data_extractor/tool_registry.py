from collections.abc import Callable
from typing import Any

from src.pdf_data_extractor.tool_schemas import (
    EXTRACT_GENERIC_FIELDS_TOOL,
    EXTRACT_INVOICE_FIELDS_TOOL,
    EXTRACT_RECEIPT_FIELDS_TOOL,
    EXTRACT_REPORT_FIELDS_TOOL,
    EXTRACT_RESUME_FIELDS_TOOL,
)
from src.pdf_data_extractor.tools import (
    classify_document,
    extract_generic_fields,
    extract_invoice_fields,
    extract_receipt_fields,
    extract_report_fields,
    extract_resume_fields,
)

ToolFunction = Callable[..., Any]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "classify_document": classify_document,
    "extract_invoice_fields": extract_invoice_fields,
    "extract_resume_fields": extract_resume_fields,
    "extract_receipt_fields": extract_receipt_fields,
    "extract_report_fields": extract_report_fields,
    "extract_generic_fields": extract_generic_fields,
}

SPECIALIZED_TOOL_NAMES: dict[str, str] = {
    "invoice": "extract_invoice_fields",
    "resume": "extract_resume_fields",
    "receipt": "extract_receipt_fields",
    "report": "extract_report_fields",
    "generic": "extract_generic_fields",
}

SPECIALIZED_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "invoice": EXTRACT_INVOICE_FIELDS_TOOL,
    "resume": EXTRACT_RESUME_FIELDS_TOOL,
    "receipt": EXTRACT_RECEIPT_FIELDS_TOOL,
    "report": EXTRACT_REPORT_FIELDS_TOOL,
    "generic": EXTRACT_GENERIC_FIELDS_TOOL,
}
