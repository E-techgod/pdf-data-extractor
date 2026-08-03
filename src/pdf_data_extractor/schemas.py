from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal[
    "invoice",
    "resume",
    "receipt",
    "report",
    "generic",
]

class DocumentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    reason: str = Field(min_length=1)

class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice"] = "invoice"
    invoice_number: str | None = None
    vendor: str | None = None
    customer: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    subtotal: float | None = Field(default=None, ge=0)
    tax: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)
    currency: str = "USD"


class ResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["resume"] = "resume"
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    professional_summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)


class ReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    quantity: float | None = Field(default=None, ge=0)
    amount: float | None = Field(default=None, ge=0)


class ReceiptData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["receipt"] = "receipt"
    merchant: str | None = None
    receipt_number: str | None = None
    transaction_date: str | None = None
    transaction_time: str | None = None
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, ge=0)
    tax: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)
    payment_method: str | None = None
    change_due: float | None = Field(default=None, ge=0)
    currency: str = "USD"


class ReportData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["report"] = "report"
    title: str | None = None
    author: str | None = None
    organization: str | None = None
    report_date: str | None = None
    executive_summary: str | None = None
    methodology: str | None = None
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    conclusion: str | None = None


class GenericDocumentData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["generic"] = "generic"
    title: str | None = None
    document_date: str | None = None
    author: str | None = None
    organization: str | None = None
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    document_text_excerpt: str | None = None

DocumentData = Annotated[
    InvoiceData
    | ResumeData
    | ReceiptData
    | ReportData
    | GenericDocumentData,
    Field(discriminator="document_type"),
]


class DocumentExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    data: DocumentData
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_document_type_match(
        self,
    ) -> "DocumentExtractionResult":
        if self.document_type != self.data.document_type:
            raise ValueError(
                "document_type must match data.document_type"
            )

        return self
