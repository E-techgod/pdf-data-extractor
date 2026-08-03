from collections.abc import Callable
from typing import Any

from src.pdf_data_extractor.tools import (
    classify_document,
    extract_generic_fields,
    extract_invoice_fields,
    extract_report_fields,
    extract_receipt_fields,
    extract_resume_fields,
)


ToolFunction = Callable[..., dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "classify_document": classify_document,
    "extract_invoice_fields": extract_invoice_fields,
    "extract_resume_fields": extract_resume_fields,
    "extract_receipt_fields": extract_receipt_fields,
    "extract_report_fields": extract_report_fields,
    "extract_generic_fields": extract_generic_fields,
}
