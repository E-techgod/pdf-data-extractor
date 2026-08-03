from typing import Any

CLASSIFY_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "classify_document",
        "description": (
            "Classify the provided document as invoice, resume, "
            "receipt, report, or generic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": [
                        "invoice",
                        "resume",
                        "receipt",
                        "report",
                        "generic",
                    ],
                    "description": (
                        "The single document category that best matches "
                        "the supplied document."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "A brief reason grounded in indicators found "
                        "inside the supplied document."
                    ),
                },
            },
            "required": [
                "document_type",
                "reason",
            ],
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

EXTRACT_RESUME_FIELDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_resume_fields",
        "description": (
            "Extract structured candidate information from a document "
            "that has been identified as a resume. Use only information "
            "explicitly present in the document. Never infer or invent "
            "missing details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {
                    "type": ["string", "null"],
                    "description": "Candidate's complete name.",
                },
                "email": {
                    "type": ["string", "null"],
                    "description": "Candidate's email address.",
                },
                "phone": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's phone number exactly as displayed."
                    ),
                },
                "location": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's city, state, country, or displayed location."
                    ),
                },
                "professional_summary": {
                    "type": ["string", "null"],
                    "description": (
                        "A concise summary based only on the resume content."
                    ),
                },
                "skills": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Technical and professional skills explicitly listed."
                    ),
                },
                "education": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Education entries preserving school, degree, "
                        "field, and dates when present."
                    ),
                },
                "experience": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Work experience entries preserving company, title, "
                        "dates, and responsibilities when present."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

EXTRACT_RECEIPT_FIELDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_receipt_fields",
        "description": (
            "Extract structured receipt information from a document "
            "that has been identified as a receipt. Use only values "
            "explicitly present in the document. Never infer or invent "
            "missing details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "merchant": {
                    "type": ["string", "null"],
                    "description": "Store or merchant name on the receipt.",
                },
                "receipt_number": {
                    "type": ["string", "null"],
                    "description": "Receipt, transaction, or reference number.",
                },
                "transaction_date": {
                    "type": ["string", "null"],
                    "description": (
                        "Transaction date exactly as shown in the receipt."
                    ),
                },
                "transaction_time": {
                    "type": ["string", "null"],
                    "description": (
                        "Transaction time exactly as shown in the receipt."
                    ),
                },
                "items": {
                    "type": ["array", "null"],
                    "items": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": ["string", "null"],
                                        "description": "Line item name.",
                                    },
                                    "quantity": {
                                        "type": ["number", "null"],
                                        "minimum": 0,
                                        "description": "Item quantity when shown.",
                                    },
                                    "qty": {
                                        "type": ["number", "null"],
                                        "minimum": 0,
                                        "description": "Item quantity when shown.",
                                    },
                                    "amount": {
                                        "type": ["number", "null"],
                                        "minimum": 0,
                                        "description": "Line item amount when shown.",
                                    },
                                },
                                "required": [],
                                "additionalProperties": False,
                            },
                        ],
                    },
                    "description": (
                        "Line items explicitly listed on the receipt."
                    ),
                },
                "subtotal": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Subtotal before tax and final charges.",
                },
                "tax": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Tax amount shown on the receipt.",
                },
                "total": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Final total charged.",
                },
                "payment_method": {
                    "type": ["string", "null"],
                    "description": "Payment method exactly as shown.",
                },
                "change_due": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "description": "Change returned to the customer.",
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

EXTRACT_REPORT_FIELDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_report_fields",
        "description": (
            "Extract structured report information from a document "
            "that has been identified as a report. Use only values "
            "explicitly present in the document. Never infer or invent "
            "missing details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": ["string", "null"],
                    "description": "Report title exactly as shown.",
                },
                "author": {
                    "type": ["string", "null"],
                    "description": "Author name exactly as shown.",
                },
                "organization": {
                    "type": ["string", "null"],
                    "description": "Organization associated with the report.",
                },
                "report_date": {
                    "type": ["string", "null"],
                    "description": "Report date exactly as shown.",
                },
                "executive_summary": {
                    "type": ["string", "null"],
                    "description": (
                        "Executive summary text explicitly present in the report."
                    ),
                },
                "methodology": {
                    "type": ["string", "null"],
                    "description": (
                        "Methodology section text explicitly present in the report."
                    ),
                },
                "findings": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": "Findings explicitly listed in the report.",
                },
                "recommendations": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Recommendations explicitly listed in the report."
                    ),
                },
                "conclusion": {
                    "type": ["string", "null"],
                    "description": "Conclusion text explicitly present in the report.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

EXTRACT_GENERIC_FIELDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_generic_fields",
        "description": (
            "Extract structured information from a document that has been "
            "identified as a generic document when it does not fit invoice, "
            "resume, receipt, or report patterns. Use only values explicitly "
            "present in the document. Never infer or invent missing details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": ["string", "null"],
                    "description": "Document title exactly as shown.",
                },
                "document_date": {
                    "type": ["string", "null"],
                    "description": "Document date exactly as shown.",
                },
                "author": {
                    "type": ["string", "null"],
                    "description": "Author name exactly as shown.",
                },
                "organization": {
                    "type": ["string", "null"],
                    "description": "Organization associated with the document.",
                },
                "summary": {
                    "type": ["string", "null"],
                    "description": (
                        "Short summary based only on the explicit document content."
                    ),
                },
                "key_points": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Key points explicitly present in the document."
                    ),
                },
                "document_text_excerpt": {
                    "type": ["string", "null"],
                    "description": (
                        "Short excerpt copied or condensed from the document text."
                    ),
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
    EXTRACT_RESUME_FIELDS_TOOL,
    EXTRACT_RECEIPT_FIELDS_TOOL,
    EXTRACT_REPORT_FIELDS_TOOL,
    EXTRACT_GENERIC_FIELDS_TOOL,
]

SPECIALIZED_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "invoice": EXTRACT_INVOICE_FIELDS_TOOL,
    "resume": EXTRACT_RESUME_FIELDS_TOOL,
    "receipt": EXTRACT_RECEIPT_FIELDS_TOOL,
    "report": EXTRACT_REPORT_FIELDS_TOOL,
    "generic": EXTRACT_GENERIC_FIELDS_TOOL,
}