from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel

from src.pdf_data_extractor.schemas import DocumentExtractionResult


@dataclass
class EvaluationCaseResult:
    case_id: str
    expected_type: str
    predicted_type: str
    classification_correct: bool
    required_fields_found: int
    required_fields_total: int
    completeness_score: float
    exact_match_fields: int
    expected_match_fields: int
    field_accuracy: float


_PATH_SEGMENT_PATTERN = re.compile(
    r"([^[.\]]+)|\[(\d+)\]"
)
_EXACT_STRING_FIELDS = {
    "invoice_number",
    "receipt_number",
    "invoice_date",
    "due_date",
    "transaction_date",
    "transaction_time",
    "report_date",
    "document_date",
    "date",
    "time",
    "currency",
}


def get_nested_value(
    data: dict[str, Any],
    path: str,
) -> Any:
    current: Any = data

    for segment in path.split("."):
        for name_token, index_token in _PATH_SEGMENT_PATTERN.findall(
            segment
        ):
            if name_token:
                if not isinstance(current, dict):
                    return None

                current = current.get(name_token)
            else:
                if (
                    not isinstance(current, list)
                    or current is None
                ):
                    return None

                index = int(index_token)

                if index >= len(current):
                    return None

                current = current[index]

            if current is None:
                return None

    return current


def _value_is_present(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return True


def calculate_completeness(
    data: dict[str, Any],
    required_fields: list[str],
) -> tuple[int, int, float]:
    if not required_fields:
        return 0, 0, 1.0

    found = 0

    for field_name in required_fields:
        value = get_nested_value(data, field_name)

        if _value_is_present(value):
            found += 1

    total = len(required_fields)
    score = found / total

    return found, total, score


def _field_name_from_path(path: str) -> str:
    leaf = path.split(".")[-1]
    return leaf.split("[", maxsplit=1)[0]


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_phone(value: str) -> str:
    return "".join(
        character for character in value if character.isdigit()
    )


def _is_phone_path(path: str) -> bool:
    return _field_name_from_path(path) == "phone"


def _is_exact_string_path(path: str) -> bool:
    return _field_name_from_path(path) in _EXACT_STRING_FIELDS


def values_match(
    actual: Any,
    expected: Any,
    *,
    field_path: str = "",
    float_tolerance: float = 0.01,
) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= float_tolerance
        except (TypeError, ValueError):
            return False

    if isinstance(actual, str) and isinstance(expected, str):
        if _is_phone_path(field_path):
            return _normalize_phone(actual) == _normalize_phone(
                expected
            )

        normalized_actual = _normalize_whitespace(actual)
        normalized_expected = _normalize_whitespace(expected)

        if _is_exact_string_path(field_path):
            return normalized_actual == normalized_expected

        return (
            normalized_actual.casefold()
            == normalized_expected.casefold()
        )

    return actual == expected


def calculate_field_accuracy(
    actual_data: dict[str, Any],
    expected_data: dict[str, Any],
) -> tuple[int, int, float]:
    if not expected_data:
        return 0, 0, 1.0

    matches = 0

    for field_name, expected_value in expected_data.items():
        actual_value = get_nested_value(
            actual_data,
            field_name,
        )

        if values_match(
            actual_value,
            expected_value,
            field_path=field_name,
        ):
            matches += 1

    total = len(expected_data)
    score = matches / total

    return matches, total, score


def _coerce_mapping(
    data: dict[str, Any] | BaseModel,
) -> dict[str, Any]:
    if isinstance(data, BaseModel):
        return data.model_dump()

    return data


def evaluate_result(
    *,
    case_id: str,
    expected_type: str,
    expected_data: dict[str, Any],
    required_fields: list[str],
    result: DocumentExtractionResult,
) -> EvaluationCaseResult:
    result_data = _coerce_mapping(result.data)

    found, required_total, completeness = calculate_completeness(
        result_data,
        required_fields,
    )

    matched, expected_total, field_accuracy = calculate_field_accuracy(
        result_data,
        expected_data,
    )

    return EvaluationCaseResult(
        case_id=case_id,
        expected_type=expected_type,
        predicted_type=result.document_type,
        classification_correct=(
            result.document_type == expected_type
        ),
        required_fields_found=found,
        required_fields_total=required_total,
        completeness_score=completeness,
        exact_match_fields=matched,
        expected_match_fields=expected_total,
        field_accuracy=field_accuracy,
    )
