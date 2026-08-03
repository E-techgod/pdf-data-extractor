from typing import Any, Literal, TypedDict

from src.pdf_data_extractor.schemas import (
    DocumentExtractionResult,
    GenericDocumentData,
    InvoiceData,
    ReportData,
    ReceiptData,
    ReceiptItem,
    ResumeData,
)


DocumentType = Literal[
    "invoice",
    "resume",
    "receipt",
    "report",
    "generic",
]


class DocumentClassification(TypedDict):
    document_type: DocumentType
    reason: str


def classify_document(document_text: str) -> DocumentClassification:

    """
    Classify a document using simple keyword rules.

    This is the local Python function that the LLM will eventually call.
    The function must remain deterministic and independent from the LLM API.
    """
    normalized_text = document_text.strip().lower()

    if not normalized_text:
        raise ValueError("document_text cannot be empty")

    keyword_groups: dict[DocumentType, tuple[str, ...]] = {
        "invoice": (
            "invoice number",
            "invoice #",
            "bill to",
            "amount due",
            "payment due",
        ),
        "resume": (
            "work experience",
            "professional experience",
            "education",
            "skills",
            "employment history",
        ),
        "receipt": (
            "receipt",
            "subtotal",
            "sales tax",
            "change due",
            "payment method",
        ),
        "report": (
            "executive summary",
            "findings",
            "methodology",
            "recommendations",
            "conclusion",
        ),
    }

    scores: dict[DocumentType, int] = {
        document_type: sum(
            keyword in normalized_text
            for keyword in keywords
        )
        for document_type, keywords in keyword_groups.items()
    }

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        return {
            "document_type": "generic",
            "reason": "No recognized document-specific keywords were found.",
        }

    return {
        "document_type": best_type,
        "reason": (
            f"Detected {scores[best_type]} keyword match(es) "
            f"associated with {best_type} documents."
        ),
    }

def extract_invoice_fields(
    invoice_number: str | None = None,
    vendor: str | None = None,
    customer: str | None = None,
    invoice_date: str | None = None,
    due_date: str | None = None,
    subtotal: float | None = None,
    tax: float | None = None,
    total: float | None = None,
    currency: str = "USD",
) -> DocumentExtractionResult:
    """
    Validate and return structured fields extracted from an invoice.

    The LLM extracts the values from the document.
    This Python function validates and normalizes those values.
    """
    invoice = InvoiceData(
        invoice_number=invoice_number,
        vendor=vendor,
        customer=customer,
        invoice_date=invoice_date,
        due_date=due_date,
        subtotal=subtotal,
        tax=tax,
        total=total,
        currency=currency.upper(),
    )

    return DocumentExtractionResult(
        document_type="invoice",
        data=invoice,
    )

def extract_resume_fields(
    full_name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    professional_summary: str | None = None,
    skills: list[str] | None = None,
    education: list[str] | None = None,
    experience: list[str] | None = None,
) -> DocumentExtractionResult:
    """
    Validate and return structured fields extracted from a resume.

    The LLM identifies the resume information.
    This function validates and normalizes the result.
    """
    resume = ResumeData(
        full_name=full_name,
        email=email,
        phone=phone,
        location=location,
        professional_summary=professional_summary,
        skills=skills or [],
        education=education or [],
        experience=experience or [],
    )

    return DocumentExtractionResult(
        document_type="resume",
        data=resume,
    )


def extract_receipt_fields(
    merchant: str | None = None,
    receipt_number: str | None = None,
    transaction_date: str | None = None,
    transaction_time: str | None = None,
    items: list[dict[str, Any] | str] | None = None,
    subtotal: float | None = None,
    tax: float | None = None,
    total: float | None = None,
    payment_method: str | None = None,
    change_due: float | None = None,
    currency: str = "USD",
) -> DocumentExtractionResult:
    """
    Validate and return structured fields extracted from a receipt.

    The LLM extracts the values from the document.
    This Python function validates and normalizes those values.
    """
    normalized_items = [
        ReceiptItem(name=item)
        if isinstance(item, str)
        else ReceiptItem.model_validate(
            {
                "name": item.get("name"),
                "quantity": item.get(
                    "quantity",
                    item.get("qty"),
                ),
                "amount": item.get("amount"),
            }
        )
        for item in (items or [])
    ]

    receipt = ReceiptData(
        merchant=merchant,
        receipt_number=receipt_number,
        transaction_date=transaction_date,
        transaction_time=transaction_time,
        items=normalized_items,
        subtotal=subtotal,
        tax=tax,
        total=total,
        payment_method=payment_method,
        change_due=change_due,
        currency=currency.upper(),
    )

    return DocumentExtractionResult(
        document_type="receipt",
        data=receipt,
    )


def extract_report_fields(
    title: str | None = None,
    author: str | None = None,
    organization: str | None = None,
    report_date: str | None = None,
    executive_summary: str | None = None,
    methodology: str | None = None,
    findings: list[str] | None = None,
    recommendations: list[str] | None = None,
    conclusion: str | None = None,
) -> DocumentExtractionResult:
    """
    Validate and return structured fields extracted from a report.

    The LLM extracts the values from the document.
    This Python function validates and normalizes those values.
    """
    report = ReportData(
        title=title,
        author=author,
        organization=organization,
        report_date=report_date,
        executive_summary=executive_summary,
        methodology=methodology,
        findings=findings or [],
        recommendations=recommendations or [],
        conclusion=conclusion,
    )

    return DocumentExtractionResult(
        document_type="report",
        data=report,
    )


def extract_generic_fields(
    title: str | None = None,
    document_date: str | None = None,
    author: str | None = None,
    organization: str | None = None,
    summary: str | None = None,
    key_points: list[str] | None = None,
    document_text_excerpt: str | None = None,
) -> DocumentExtractionResult:
    """
    Validate and return structured fields extracted from a generic document.

    The LLM extracts the values from the document.
    This Python function validates and normalizes those values.
    """
    generic_document = GenericDocumentData(
        title=title,
        document_date=document_date,
        author=author,
        organization=organization,
        summary=summary,
        key_points=key_points or [],
        document_text_excerpt=document_text_excerpt,
    )

    return DocumentExtractionResult(
        document_type="generic",
        data=generic_document,
    )
