"""
Chaos / adversarial-input tests for src/pdf_data_extractor/tools.py.

These do NOT test the happy path (already covered by tests/test_tools.py).
Every test here feeds tools.py a malformed-but-plausible payload of the kind
a real LLM tool call can produce (JSON `null`, JSON `true`/`false`, a bare
number, a nested list) and records the ACTUAL resulting behavior. Several
of these currently document real defects rather than confirm correctness;
each comment explains the concrete risk.
"""

import json

import pytest
from pydantic import ValidationError

from src.pdf_data_extractor.tools import (
    extract_invoice_fields,
    extract_receipt_fields,
    extract_resume_fields,
)


def test_invoice_currency_none_crashes_with_unhandled_attributeerror() -> None:
    # Risk: the LLM tool-call JSON schema marks every field optional and an
    # LLM can legally emit `"currency": null` (e.g. when it "doesn't know"
    # the currency). `currency.upper()` in extract_invoice_fields has no
    # None-guard, so this raises a raw AttributeError instead of a Pydantic
    # ValidationError. agent.py's _execute_tool_call only catches TypeError,
    # so this exception is NOT converted into the clean
    # "Invalid arguments for tool" ValueError -- it crashes the whole
    # extraction request with an unhandled, uncontextualized error.
    with pytest.raises(AttributeError):
        extract_invoice_fields(
            vendor="Example Services LLC",
            currency=None,
        )


def test_receipt_currency_none_crashes_with_unhandled_attributeerror() -> None:
    # Same defect as the invoice extractor, duplicated in the receipt path.
    with pytest.raises(AttributeError):
        extract_receipt_fields(
            merchant="HEB",
            currency=None,
        )


@pytest.mark.parametrize(
    "bad_item",
    [42, 3.14, True, None, ["nested", "list"], {1, 2, 3}],
)
def test_resume_experience_non_dict_item_crashes_instead_of_validating(
    bad_item: object,
) -> None:
    # Risk: extract_resume_fields normalizes `experience` items with
    # `item.get("company")` directly whenever isinstance(item, str) is
    # False -- with no dict-type check first. Any non-str, non-dict item
    # (int, float, bool, None, list, set) raises AttributeError from `.get`
    # before Pydantic ever gets a chance to validate the shape. This is
    # inconsistent with the sibling `education` field, which instead calls
    # `ResumeEducation.model_validate(item)` directly and correctly raises a
    # catchable pydantic ValidationError for the exact same class of bad
    # input. The asymmetry means malformed `experience` entries crash
    # ungracefully while malformed `education` entries fail cleanly.
    with pytest.raises(AttributeError):
        extract_resume_fields(experience=[bad_item])


@pytest.mark.parametrize(
    "bad_item",
    [42, 3.14, True, None, ["nested", "list"]],
)
def test_receipt_items_non_dict_item_crashes_instead_of_validating(
    bad_item: object,
) -> None:
    # Same class of bug as the resume `experience` case, in
    # extract_receipt_fields: `item.get("name")` is called unconditionally
    # once isinstance(item, str) is False, so a malformed line item (e.g.
    # the LLM emits a bare number where an object was expected) crashes
    # with AttributeError instead of a clean, catchable ValidationError.
    with pytest.raises(AttributeError):
        extract_receipt_fields(items=[bad_item])


def test_receipt_quantity_explicit_null_does_not_fall_back_to_qty() -> None:
    # Risk / logical fallacy: the code does
    #     item.get("quantity", item.get("qty"))
    # `dict.get`'s default is only used when the key is MISSING, never when
    # the key is present with value None. Since the default expression
    # `item.get("qty")` is evaluated eagerly regardless, a caller could
    # reasonably assume "quantity falls back to qty when quantity is
    # unavailable" -- but if the LLM emits `{"quantity": null, "qty": 5}`
    # (a very plausible shape when a receipt shows "qty" but the model also
    # echoes a null "quantity" key from the schema), the real value 5 is
    # silently discarded and quantity ends up None. No error, no warning --
    # just quietly wrong/missing data.
    result = extract_receipt_fields(
        items=[{"name": "Milk", "quantity": None, "qty": 5}],
    )

    assert result.data.items[0].quantity is None


def test_invoice_total_nan_is_rejected_by_validation() -> None:
    # Boundary control case: confirms NaN correctly fails the `ge=0`
    # constraint (NaN comparisons are always False, so `nan >= 0` is
    # False and pydantic-core rejects it). Paired with the Infinity test
    # below to show the *inconsistent* handling of the two "exotic" float
    # values that ge=0 is supposed to gate.
    with pytest.raises(ValidationError):
        extract_invoice_fields(
            vendor="Example Services LLC",
            total=float("nan"),
        )


def test_invoice_total_infinity_passes_validation_then_silently_becomes_json_null() -> (
    None
):
    # Risk: unlike NaN, `float("inf") >= 0` is True, so `ge=0` does NOT
    # reject Infinity -- an LLM (or any caller) can smuggle a total of
    # Infinity straight through Pydantic validation. Worse: main.py's only
    # output path is `result.model_dump_json(indent=2)`, and pydantic's
    # JSON serializer silently converts Infinity to JSON `null` rather than
    # raising or emitting a non-standard `Infinity` token. The in-memory
    # object and the JSON a human/downstream system actually reads
    # disagree: Python sees total == inf, but the printed/consumed JSON
    # shows "total": null, indistinguishable from "no total was found".
    # This is a silent data-corruption path, not a crash.
    result = extract_invoice_fields(
        vendor="Example Services LLC",
        total=float("inf"),
    )

    assert result.data.total == float("inf")

    dumped = json.loads(result.model_dump_json())
    assert dumped["data"]["total"] is None


def test_invoice_total_boolean_is_silently_coerced_to_float() -> None:
    # Type coercion risk: Python's bool is an int subclass, and Pydantic's
    # lenient float validation accepts True/False as 1.0/0.0 without
    # complaint. If a malformed LLM tool call emits `"total": true` (e.g.
    # confusing a numeric field with a boolean one, or a JSON parsing slip
    # upstream), this is *silently* accepted as a real monetary total of
    # 1.0 instead of being rejected as a type mismatch.
    result = extract_invoice_fields(
        vendor="Example Services LLC",
        total=True,
    )

    assert result.data.total == 1.0
    assert isinstance(result.data.total, float)


def test_receipt_quantity_boolean_is_silently_coerced_to_float() -> None:
    # Same bool -> float coercion trap as above, exercised through the
    # nested ReceiptItem model reached via extract_receipt_fields' item
    # normalization path.
    result = extract_receipt_fields(
        items=[{"name": "Milk", "quantity": False, "amount": 2.5}],
    )

    assert result.data.items[0].quantity == 0.0
    assert isinstance(result.data.items[0].quantity, float)
    assert not isinstance(result.data.items[0].quantity, bool)
