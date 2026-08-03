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
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Technical and professional skills explicitly listed."
                    ),
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Education entries preserving school, degree, "
                        "field, and dates when present."
                    ),
                },
                "experience": {
                    "type": "array",
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
                    "type": "array",
                    "items": {
                        "type": "string",
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

TOOLS: list[dict[str, Any]] = [
    CLASSIFY_DOCUMENT_TOOL,
    EXTRACT_INVOICE_FIELDS_TOOL,
    EXTRACT_RESUME_FIELDS_TOOL,
    EXTRACT_RECEIPT_FIELDS_TOOL,
]
