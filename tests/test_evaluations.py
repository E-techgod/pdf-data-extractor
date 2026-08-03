from src.pdf_data_extractor.evaluation import (
    calculate_completeness,
    calculate_field_accuracy,
    evaluate_result,
    get_nested_value,
    values_match,
)
from src.pdf_data_extractor.schemas import (
    DocumentExtractionResult,
    InvoiceData,
)


def test_complete_invoice_scores_one() -> None:
    data = {
        "invoice_number": "INV-001",
        "vendor": "Example LLC",
        "customer": "Customer LLC",
        "total": 100.0,
    }

    required_fields = [
        "invoice_number",
        "vendor",
        "customer",
        "total",
    ]

    found, total, score = calculate_completeness(
        data,
        required_fields,
    )

    assert found == 4
    assert total == 4
    assert score == 1.0


def test_missing_fields_reduce_completeness() -> None:
    data = {
        "invoice_number": "INV-001",
        "vendor": None,
        "customer": "",
        "total": 100.0,
    }

    required_fields = [
        "invoice_number",
        "vendor",
        "customer",
        "total",
    ]

    found, total, score = calculate_completeness(
        data,
        required_fields,
    )

    assert found == 2
    assert total == 4
    assert score == 0.5


def test_field_accuracy_is_case_insensitive() -> None:
    actual = {
        "vendor": "NORTHSTAR AUTOMATION LLC",
    }

    expected = {
        "vendor": "Northstar Automation LLC",
    }

    matched, total, score = calculate_field_accuracy(
        actual,
        expected,
    )

    assert matched == 1
    assert total == 1
    assert score == 1.0


def test_field_accuracy_allows_float_tolerance() -> None:
    actual = {
        "total": 2024.279,
    }

    expected = {
        "total": 2024.28,
    }

    matched, total, score = calculate_field_accuracy(
        actual,
        expected,
    )

    assert matched == 1
    assert total == 1
    assert score == 1.0


def test_evaluate_result_detects_wrong_classification() -> None:
    result = DocumentExtractionResult(
        document_type="resume",
        data={},
        warnings=[],
    )

    evaluation = evaluate_result(
        case_id="invoice_001",
        expected_type="invoice",
        expected_data={},
        required_fields=[],
        result=result,
    )

    assert evaluation.classification_correct is False
    assert evaluation.expected_type == "invoice"
    assert evaluation.predicted_type == "resume"


def test_evaluate_result_accepts_pydantic_document_data() -> None:
    result = DocumentExtractionResult(
        document_type="invoice",
        data=InvoiceData(
            invoice_number="INV-001",
            vendor="Example LLC",
            total=100.0,
        ),
        warnings=[],
    )

    evaluation = evaluate_result(
        case_id="invoice_001",
        expected_type="invoice",
        expected_data={
            "invoice_number": "INV-001",
            "vendor": "Example LLC",
            "total": 100.0,
        },
        required_fields=[
            "invoice_number",
            "vendor",
            "total",
        ],
        result=result,
    )

    assert evaluation.classification_correct is True
    assert evaluation.required_fields_found == 3
    assert evaluation.exact_match_fields == 3


def test_get_nested_value_reads_dictionary_path() -> None:
    data = {
        "education": [
            {
                "school": "University of Houston",
            }
        ]
    }

    assert (
        get_nested_value(data, "education[0].school")
        == "University of Houston"
    )


def test_get_nested_value_reads_list_index_path() -> None:
    data = {
        "experience": [
            {
                "responsibilities": [
                    "Built AI workflows.",
                ]
            }
        ]
    }

    assert (
        get_nested_value(
            data,
            "experience[0].responsibilities[0]",
        )
        == "Built AI workflows."
    )


def test_get_nested_value_returns_none_for_missing_nested_key() -> None:
    data = {
        "education": [
            {
                "school": "University of Houston",
            }
        ]
    }

    assert (
        get_nested_value(data, "education[0].degree")
        is None
    )


def test_get_nested_value_returns_none_for_out_of_range_index() -> None:
    data = {
        "education": [
            {
                "school": "University of Houston",
            }
        ]
    }

    assert (
        get_nested_value(data, "education[1].school")
        is None
    )


def test_completeness_supports_nested_fields() -> None:
    data = {
        "education": [
            {
                "school": "University of Houston",
                "degree": "B.S.",
            }
        ]
    }

    found, total, score = calculate_completeness(
        data,
        [
            "education[0].school",
            "education[0].degree",
            "education[0].field",
        ],
    )

    assert found == 2
    assert total == 3
    assert score == 2 / 3


def test_field_accuracy_supports_nested_fields() -> None:
    actual = {
        "education": [
            {
                "school": "University of Houston",
                "degree": "Bachelor of Science",
            }
        ]
    }

    expected = {
        "education[0].school": "University of Houston",
        "education[0].degree": "Bachelor of Science",
        "education[0].field": "Computer Science",
    }

    matched, total, score = calculate_field_accuracy(
        actual,
        expected,
    )

    assert matched == 2
    assert total == 3
    assert score == 2 / 3


def test_field_accuracy_normalizes_whitespace() -> None:
    actual = {
        "vendor": "Northstar   Automation   LLC",
    }

    expected = {
        "vendor": " Northstar Automation LLC ",
    }

    matched, total, score = calculate_field_accuracy(
        actual,
        expected,
    )

    assert matched == 1
    assert total == 1
    assert score == 1.0


def test_values_match_normalizes_phone_numbers() -> None:
    assert (
        values_match(
            "(832) 555-0198",
            "8325550198",
            field_path="phone",
        )
        is True
    )


def test_values_match_keeps_identifier_comparisons_exact() -> None:
    assert (
        values_match(
            "inv-001",
            "INV-001",
            field_path="invoice_number",
        )
        is False
    )


def test_empty_lists_and_dicts_count_as_missing() -> None:
    data = {
        "skills": [],
        "metadata": {},
    }

    found, total, score = calculate_completeness(
        data,
        ["skills", "metadata"],
    )

    assert found == 0
    assert total == 2
    assert score == 0.0


def test_no_required_fields_returns_complete_score() -> None:
    found, total, score = calculate_completeness(
        {},
        [],
    )

    assert found == 0
    assert total == 0
    assert score == 1.0
