import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.pdf_data_extractor.agent import analyze_document_with_groq
from src.pdf_data_extractor.evaluation import (
    evaluate_result,
    get_nested_value,
    values_match,
)

DATASET_PATH = Path("evals/golden_dataset.json")
CASE_IDS: set[str] | None = None


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _coerce_result_data(
    data: dict | BaseModel,
) -> dict:
    if isinstance(data, BaseModel):
        return data.model_dump()

    return data


def _selected_cases(
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not CASE_IDS:
        return dataset

    return [case for case in dataset if case["id"] in CASE_IDS]


def main() -> None:
    dataset = _selected_cases(load_dataset())
    case_results = []

    for case in dataset:
        print(f"\nRunning: {case['id']}")

        result = analyze_document_with_groq(case["text"])
        result_data = _coerce_result_data(result.data)

        evaluation = evaluate_result(
            case_id=case["id"],
            expected_type=case["document_type"],
            expected_data=case["expected_data"],
            required_fields=case["required_fields"],
            result=result,
        )

        case_results.append(evaluation)

        print(
            f"Classification: "
            f"{evaluation.predicted_type} "
            f"(expected {evaluation.expected_type})"
        )

        print(f"Classification correct: {evaluation.classification_correct}")

        print(
            f"Completeness: "
            f"{evaluation.required_fields_found}/"
            f"{evaluation.required_fields_total} "
            f"({evaluation.completeness_score:.1%})"
        )

        print(
            f"Field accuracy: "
            f"{evaluation.exact_match_fields}/"
            f"{evaluation.expected_match_fields} "
            f"({evaluation.field_accuracy:.1%})"
        )

        print("\nPredicted data:")
        print(
            json.dumps(
                result_data,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nExpected checked fields:")
        print(
            json.dumps(
                case["expected_data"],
                indent=2,
                ensure_ascii=False,
            )
        )

        mismatches_found = False
        for field_name, expected_value in case["expected_data"].items():
            actual_value = get_nested_value(
                result_data,
                field_name,
            )

            if values_match(
                actual_value,
                expected_value,
                field_path=field_name,
            ):
                continue

            mismatches_found = True
            print(f"\nMismatch: {field_name}")
            print(f"Expected: {expected_value!r}")
            print(f"Actual: {actual_value!r}")

        if not mismatches_found:
            print("\nNo field mismatches.")

    classification_accuracy = sum(
        result.classification_correct for result in case_results
    ) / len(case_results)

    average_completeness = sum(
        result.completeness_score for result in case_results
    ) / len(case_results)

    average_field_accuracy = sum(
        result.field_accuracy for result in case_results
    ) / len(case_results)

    print("\n=== Evaluation Summary ===")
    print(f"Classification accuracy: {classification_accuracy:.1%}")
    print(f"Average completeness: {average_completeness:.1%}")
    print(f"Average field accuracy: {average_field_accuracy:.1%}")


if __name__ == "__main__":
    main()
