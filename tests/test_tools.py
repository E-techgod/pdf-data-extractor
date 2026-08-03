import pytest
from pydantic import ValidationError

from src.pdf_data_extractor.agent import _build_assistant_tool_message
from src.pdf_data_extractor.schemas import (
    DocumentClassification,
    DocumentExtractionResult,
    GenericDocumentData,
    InvoiceData,
    ReceiptItem,
    ReceiptData,
    ReportData,
    ResumeEducation,
    ResumeExperience,
    ResumeData,
)
from src.pdf_data_extractor.tools import (
    classify_document,
    extract_generic_fields,
    extract_invoice_fields,
    extract_report_fields,
    extract_receipt_fields,
    extract_resume_fields,
)


def test_classifies_invoice() -> None:
    result = classify_document(
        document_type="invoice",
        reason="Contains invoice number and amount due.",
    )

    assert result == DocumentClassification(
        document_type="invoice",
        reason="Contains invoice number and amount due.",
    )


def test_classifies_resume() -> None:
    result = classify_document(
        document_type="resume",
        reason="Contains education, skills, and experience.",
    )

    assert result.document_type == "resume"


def test_classifies_receipt() -> None:
    result = classify_document(
        document_type="receipt",
        reason="Contains subtotal and payment method.",
    )

    assert result.document_type == "receipt"


def test_classifies_report() -> None:
    result = classify_document(
        document_type="report",
        reason="Contains executive summary and findings.",
    )

    assert result.document_type == "report"


def test_classifies_unknown_document_as_generic() -> None:
    result = classify_document(
        document_type="generic",
        reason="Does not match a specialized type.",
    )

    assert result.document_type == "generic"


def test_rejects_empty_document() -> None:
    with pytest.raises(ValidationError):
        classify_document(
            document_type="invoice",
            reason="",
        )


def test_rejects_invalid_document_type() -> None:
    with pytest.raises(ValidationError):
        classify_document(
            document_type="memo",
            reason="Unsupported type.",
        )


def test_build_assistant_tool_message_omits_sdk_only_fields() -> None:
    class FakeFunction:
        name = "classify_document"
        arguments = '{"document_type":"invoice","reason":"sample"}'

    class FakeToolCall:
        id = "call_123"
        type = "function"
        function = FakeFunction()

    class FakeAssistantMessage:
        content = None
        tool_calls = [FakeToolCall()]
        annotations = [{"unsupported": True}]

    result = _build_assistant_tool_message(
        FakeAssistantMessage()
    )

    assert result == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "classify_document",
                    "arguments": (
                        '{"document_type":"invoice","reason":"sample"}'
                    ),
                },
            }
        ],
    }


def test_extracts_invoice_fields() -> None:
    result = extract_invoice_fields(
        invoice_number="INV-1001",
        vendor="Example Services LLC",
        customer="Elias Arellano",
        invoice_date="August 1, 2026",
        due_date="August 15, 2026",
        subtotal=500.00,
        tax=41.25,
        total=541.25,
        currency="usd",
    )

    assert isinstance(result, DocumentExtractionResult)
    assert result.document_type == "invoice"
    assert isinstance(result.data, InvoiceData)
    assert result.data.invoice_number == "INV-1001"
    assert result.data.vendor == "Example Services LLC"
    assert result.data.total == 541.25
    assert result.data.currency == "USD"


def test_invoice_allows_missing_fields() -> None:
    result = extract_invoice_fields(
        vendor="Example Services LLC",
        total=125.00,
    )

    assert result.data.vendor == "Example Services LLC"
    assert result.data.invoice_number is None
    assert result.data.total == 125.00


def test_invoice_rejects_negative_total() -> None:
    with pytest.raises(ValidationError):
        extract_invoice_fields(
            vendor="Example Services LLC",
            total=-50.00,
        )


def test_extracts_resume_fields() -> None:
    result = extract_resume_fields(
        full_name="Elias Arellano Campos",
        email="elias@example.com",
        phone="832-555-0100",
        location="Houston, Texas",
        professional_summary=(
            "Computer Science graduate focused on applied AI engineering."
        ),
        skills=[
            "Python",
            "SQL",
            "PyTorch",
            "FastAPI",
        ],
        education=[
            {
                "school": "University of Houston",
                "degree": "B.S.",
                "field": "Computer Science",
                "graduation_date": "May 2026",
            }
        ],
        experience=[
            {
                "company": "American Smart Business LLC",
                "title": "AI Systems Implementation Associate",
                "dates": "2026",
                "responsibilities": [
                    "Built AI workflows."
                ],
            }
        ],
    )

    assert result.document_type == "resume"
    assert isinstance(result.data, ResumeData)
    assert result.data.full_name == "Elias Arellano Campos"
    assert result.data.email == "elias@example.com"
    assert "Python" in result.data.skills
    assert result.data.education == [
        ResumeEducation(
            school="University of Houston",
            degree="B.S.",
            field="Computer Science",
            graduation_date="May 2026",
        )
    ]
    assert result.data.experience == [
        ResumeExperience(
            company="American Smart Business LLC",
            title="AI Systems Implementation Associate",
            dates="2026",
            responsibilities=["Built AI workflows."],
        )
    ]


def test_resume_allows_missing_fields() -> None:
    result = extract_resume_fields(
        full_name="Elias Arellano Campos",
        skills=["Python"],
    )

    assert result.data.full_name == "Elias Arellano Campos"
    assert result.data.email is None
    assert result.data.education == []


def test_resume_does_not_share_mutable_lists() -> None:
    first_result = extract_resume_fields(
        full_name="Candidate One",
    )
    second_result = extract_resume_fields(
        full_name="Candidate Two",
    )

    assert isinstance(first_result.data, ResumeData)
    assert isinstance(second_result.data, ResumeData)

    first_result.data.skills.append("Python")

    assert second_result.data.skills == []


def test_extracts_receipt_fields() -> None:
    result = extract_receipt_fields(
        merchant="HEB",
        receipt_number="RCPT-1001",
        transaction_date="August 1, 2026",
        transaction_time="14:35",
        items=[
            {
                "name": "Milk",
                "quantity": 1,
                "amount": 4.25,
            },
            {
                "name": "Bread",
                "quantity": 1,
                "amount": 3.15,
            },
        ],
        subtotal=12.50,
        tax=1.03,
        total=13.53,
        payment_method="Visa",
        change_due=0.00,
        currency="usd",
    )

    assert result.document_type == "receipt"
    assert isinstance(result.data, ReceiptData)
    assert result.data.merchant == "HEB"
    assert result.data.receipt_number == "RCPT-1001"
    assert result.data.total == 13.53
    assert result.data.payment_method == "Visa"
    assert result.data.currency == "USD"
    assert result.data.items == [
        ReceiptItem(
            name="Milk",
            quantity=1,
            amount=4.25,
        ),
        ReceiptItem(
            name="Bread",
            quantity=1,
            amount=3.15,
        ),
    ]


def test_receipt_allows_missing_fields() -> None:
    result = extract_receipt_fields(
        merchant="HEB",
        total=13.53,
    )

    assert result.data.merchant == "HEB"
    assert result.data.receipt_number is None
    assert result.data.items == []
    assert result.data.total == 13.53


def test_receipt_rejects_negative_total() -> None:
    with pytest.raises(ValidationError):
        extract_receipt_fields(
            merchant="HEB",
            total=-13.53,
        )


def test_receipt_does_not_share_mutable_lists() -> None:
    first_result = extract_receipt_fields(
        merchant="Store One",
    )
    second_result = extract_receipt_fields(
        merchant="Store Two",
    )

    assert isinstance(first_result.data, ReceiptData)
    assert isinstance(second_result.data, ReceiptData)

    first_result.data.items.append(
        ReceiptItem(name="Milk")
    )

    assert second_result.data.items == []


def test_extracts_report_fields() -> None:
    result = extract_report_fields(
        title="Customer Retention Analysis",
        author="Elias Arellano Campos",
        organization="American Smart Business LLC",
        report_date="August 1, 2026",
        executive_summary="Retention improved during the last quarter.",
        methodology="We analyzed quarterly transaction and support data.",
        findings=[
            "Customer retention improved by 12%.",
            "Repeat purchases increased in July.",
        ],
        recommendations=[
            "Continue monitoring retention monthly.",
        ],
        conclusion="The retention strategy is producing measurable gains.",
    )

    assert result.document_type == "report"
    assert isinstance(result.data, ReportData)
    assert result.data.title == "Customer Retention Analysis"
    assert result.data.author == "Elias Arellano Campos"
    assert len(result.data.findings) == 2
    assert len(result.data.recommendations) == 1
    assert result.data.conclusion == (
        "The retention strategy is producing measurable gains."
    )


def test_report_allows_missing_fields() -> None:
    result = extract_report_fields(
        title="Customer Retention Analysis",
        findings=["Retention improved by 12%."],
    )

    assert result.data.title == "Customer Retention Analysis"
    assert result.data.author is None
    assert result.data.findings == ["Retention improved by 12%."]
    assert result.data.recommendations == []


def test_report_does_not_share_mutable_lists() -> None:
    first_result = extract_report_fields(
        title="Report One",
    )
    second_result = extract_report_fields(
        title="Report Two",
    )

    assert isinstance(first_result.data, ReportData)
    assert isinstance(second_result.data, ReportData)

    first_result.data.findings.append("Finding one")

    assert second_result.data.findings == []


def test_extracts_generic_fields() -> None:
    result = extract_generic_fields(
        title="Project Kickoff Notes",
        document_date="August 2, 2026",
        author="Elias Arellano Campos",
        organization="Austin Cohort",
        summary="Notes covering the initial kickoff discussion.",
        key_points=[
            "Team introductions completed.",
            "Project timeline was reviewed.",
        ],
        document_text_excerpt="Kickoff meeting notes for the PDF extractor project.",
    )

    assert result.document_type == "generic"
    assert isinstance(result.data, GenericDocumentData)
    assert result.data.title == "Project Kickoff Notes"
    assert result.data.document_date == "August 2, 2026"
    assert result.data.author == "Elias Arellano Campos"
    assert len(result.data.key_points) == 2


def test_generic_allows_missing_fields() -> None:
    result = extract_generic_fields(
        title="Meeting Notes",
        key_points=["Reviewed next steps."],
    )

    assert result.data.title == "Meeting Notes"
    assert result.data.author is None
    assert result.data.key_points == ["Reviewed next steps."]
    assert result.data.summary is None


def test_generic_does_not_share_mutable_lists() -> None:
    first_result = extract_generic_fields(
        title="Document One",
    )
    second_result = extract_generic_fields(
        title="Document Two",
    )

    assert isinstance(first_result.data, GenericDocumentData)
    assert isinstance(second_result.data, GenericDocumentData)

    first_result.data.key_points.append("Point one")

    assert second_result.data.key_points == []


def test_top_level_result_rejects_mismatched_data_type() -> None:
    with pytest.raises(
        ValidationError,
        match="document_type must match data.document_type",
    ):
        DocumentExtractionResult(
            document_type="invoice",
            data=GenericDocumentData(
                title="Notes",
            ),
        )
