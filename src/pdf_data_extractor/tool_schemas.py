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
            "Never invent missing information. Do not concatenate multiple "
            "semantic fields into one value. Return only the issuing entity "
            "in vendor and only the billed party in customer."
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
                    "description": (
                        "Issuing business or seller only. Do not include "
                        "customer names, addresses, service provider labels, "
                        "departments, or unrelated organizations."
                    ),
                },
                "customer": {
                    "type": ["string", "null"],
                    "description": (
                        "Billed party only, preferably the entity under "
                        "'Bill To'. Do not include addresses, service "
                        "provider labels, departments, or unrelated "
                        "organization names."
                    ),
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
            "missing details. Do not concatenate multiple semantic fields "
            "into one value. Split school, degree, field, minor, graduation "
            "date, company, title, dates, and location into their matching "
            "properties. Capture the visible resume header contact block into "
            "full_name, email, phone, and location whenever present."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's complete name from the resume header or "
                        "contact block only."
                    ),
                },
                "email": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's email address from the visible contact "
                        "block only."
                    ),
                },
                "phone": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's phone number from the visible contact "
                        "block, exactly as displayed."
                    ),
                },
                "location": {
                    "type": ["string", "null"],
                    "description": (
                        "Candidate's city, state, country, or displayed "
                        "location from the contact block only."
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
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "school": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Institution name only. Do not "
                                            "include degree, field, minor, "
                                            "or graduation date."
                                        ),
                                    },
                                    "degree": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Degree type only, such as "
                                            "'Bachelor of Science'. Do not "
                                            "include the major, field, minor, "
                                            "or phrases like 'in Computer Science'."
                                        ),
                                    },
                                    "field": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Major or field of study only, "
                                            "such as 'Computer Science'. Do not "
                                            "repeat the degree type or minor."
                                        ),
                                    },
                                    "minor": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Minor only, such as 'Mathematics'. "
                                            "Do not include the word 'Minor'."
                                        ),
                                    },
                                    "graduation_date": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Graduation date only, exactly as shown."
                                        ),
                                    },
                                },
                                "required": [],
                                "additionalProperties": False,
                            },
                        ],
                    },
                    "description": (
                        "Education entries preserving school, degree, "
                        "field, minor, and dates when present. Keep each "
                        "semantic part in its own property."
                    ),
                },
                "experience": {
                    "type": ["array", "null"],
                    "items": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "company": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Organization name only. Do not "
                                            "include title, dates, or location."
                                        ),
                                    },
                                    "title": {
                                        "type": ["string", "null"],
                                        "description": "Job title only.",
                                    },
                                    "dates": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "Employment date range only, exactly as shown."
                                        ),
                                    },
                                    "location": {
                                        "type": ["string", "null"],
                                        "description": "Location only, exactly as shown.",
                                    },
                                    "responsibilities": {
                                        "type": ["array", "null"],
                                        "items": {
                                            "type": "string",
                                        },
                                        "description": (
                                            "Responsibilities or achievements explicitly listed."
                                        ),
                                    },
                                },
                                "required": [],
                                "additionalProperties": False,
                            },
                        ],
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
