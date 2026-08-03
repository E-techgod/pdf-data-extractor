from typing import Any

CLASSIFY_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "classify_document",
        "description": (
            "Classify a document as an invoice, resume, receipt, "
            "report, or generic document using the document "
            "already provided in the conversation context."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

EXTRACT_INVOICE_FIELDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_invoice_fields",
        "description": (
            "Extract structured invoice information from a document "
            "that has already been identified as an invoice. "
            "Use null or omit a field when the value is not present. "
            "Never invent missing information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "type": ["string", "null"],
                    "description": "Invoice identifier or invoice number.",
                },
                "vendor": {
                    "type": ["string", "null"],
                    "description": "Business or person issuing the invoice.",
                },
                "customer": {
                    "type": ["string", "null"],
                    "description": "Customer or organization being billed.",
                },
                "invoice_date": {
                    "type": ["string", "null"],
                    "description": (
                        "Invoice issue date exactly as shown in the document."
                    ),
                },
                "due_date": {
                    "type": ["string", "null"],
                    "description": (
                        "Payment due date exactly as shown in the document."
                    ),
                },
                "subtotal": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Subtotal before tax and additional charges.",
                },
                "tax": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Tax amount.",
                },
                "total": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Final invoice total.",
                },
                "currency": {
                    "type": "string",
                    "description": (
                        "Three-letter currency code such as USD, MXN, or EUR."
                    ),
                    "default": "USD",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict[str, Any]] = [
    CLASSIFY_DOCUMENT_TOOL,
    EXTRACT_INVOICE_FIELDS_TOOL,
]