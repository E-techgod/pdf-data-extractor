"""
Chaos / adversarial-input tests for src/pdf_data_extractor/evaluation.py.

This module is the scoring engine for the eval harness: it decides whether
an extraction "passed." A silent scoring bug here is worse than a crash --
it makes a broken agent look healthy in a report. These tests do NOT
confirm the happy path (already covered by tests/test_evaluations.py).
Each test either (a) proves the function crashes on input the eval dataset
loader can plausibly hand it, or (b) proves it returns a *wrong but
plausible-looking* score instead of raising, which is the more dangerous
failure mode for a metrics module.
"""

import pytest

from src.pdf_data_extractor.evaluation import (
    calculate_completeness,
    calculate_field_accuracy,
    evaluate_result,
    get_nested_value,
    values_match,
)
from src.pdf_data_extractor.schemas import DocumentExtractionResult


# ---------------------------------------------------------------------------
# Boundary & Extreme Values
# ---------------------------------------------------------------------------


def test_values_match_nan_never_equals_itself() -> None:
    # Risk: `values_match` treats any float as the float-tolerance path and
    # does `abs(actual - expected) <= tolerance`. NaN's defining property
    # is that every comparison involving it -- including `<=` -- is False.
    # If a document's extracted "total" ever comes back NaN (e.g. from a
    # bad float parse upstream), the eval harness will report a mismatch
    # against literally any expected value, including NaN itself, with no
    # error raised to explain why.
    assert values_match(float("nan"), float("nan")) is False


def test_values_match_infinity_versus_infinity_fails() -> None:
    # Risk: `inf - inf` is NaN, not 0, in IEEE-754 arithmetic. So two
    # "obviously equal" infinite values silently fail the float-tolerance
    # check instead of matching or raising. A dataset with an intentionally
    # unbounded expected value (e.g. "any amount over X" modeled as inf)
    # would always be scored as wrong.
    assert values_match(float("inf"), float("inf")) is False


def test_get_nested_value_huge_bracket_index_does_not_overflow_or_crash() -> None:
    # Risk: `int(index_token)` on a regex-captured digit string has no
    # upper bound check. Python ints are arbitrary precision, so this
    # can't overflow, but it's worth locking in that a pathological index
    # from a malformed dataset path degrades to a clean None instead of
    # an OverflowError or a hang.
    data = {"items": [1, 2, 3]}
    assert get_nested_value(data, "items[999999999999999999999999]") is None


def test_calculate_completeness_handles_massive_required_fields_list() -> None:
    # Risk: nothing in calculate_completeness bounds the size of
    # required_fields. A corrupted or duplicated eval-dataset row (e.g. a
    # CSV parsing bug that repeats a header 50,000 times) should degrade
    # gracefully in linear time, not hang or blow memory.
    required_fields = [f"field_{i}" for i in range(50_000)]
    data = {f"field_{i}": "present" for i in range(0, 50_000, 2)}

    found, total, score = calculate_completeness(data, required_fields)

    assert total == 50_000
    assert found == 25_000
    assert score == 0.5


def test_empty_tuple_and_set_incorrectly_count_as_present() -> None:
    # Risk: `_value_is_present` only special-cases `list` and `dict` for
    # emptiness. An empty tuple or empty set -- both falsy, both
    # semantically "no data" -- fall through to `return True` because
    # they're neither None, str, list, nor dict. If any extractor or
    # normalization step ever hands this module a tuple/set (e.g. from a
    # `set()` dedup step), a genuinely empty field is scored as present.
    data = {"skills": (), "tags": set()}

    found, total, score = calculate_completeness(data, ["skills", "tags"])

    # Documents the bug: an empty tuple/set is NOT treated like an empty
    # list/dict, even though it is equally "no data."
    assert found == 2
    assert score == 1.0


def test_boolean_false_and_zero_count_as_present_field() -> None:
    # Risk: `_value_is_present` has no numeric/bool branch, so it falls
    # through to `return True` for anything that isn't None/str/list/dict.
    # A legitimately falsy-but-meaningful value -- `False` for a "has
    # discount" field, `0` for a "years of experience" field -- counts as
    # "present," which is almost certainly the intended behavior for 0 but
    # is easy to get wrong for bool fields with a *third* "unset" state
    # modeled as None vs False. Locking in current behavior so a future
    # refactor doesn't silently change scoring semantics.
    data = {"has_discount": False, "years_experience": 0}

    found, total, score = calculate_completeness(
        data, ["has_discount", "years_experience"]
    )

    assert found == 2
    assert score == 1.0


# ---------------------------------------------------------------------------
# Nullability & Types
# ---------------------------------------------------------------------------


def test_get_nested_value_raises_when_path_is_not_a_string() -> None:
    # Risk: the type hint says `path: str`, but nothing enforces it at
    # runtime. `path.split(".")` on a non-string (e.g. an int field-index
    # that slipped through a malformed dataset row) raises a raw
    # AttributeError instead of a clean, catchable validation error.
    with pytest.raises(AttributeError):
        get_nested_value({"a": 1}, 5)  # type: ignore[arg-type]


def test_calculate_completeness_none_required_fields_must_raise_typeerror() -> None:
    # Contract under test: `required_fields` is typed `list[str]`. A `None`
    # value (e.g. an eval dataset row where the "required_fields" JSON key
    # was accidentally `null` instead of `[]`) is not a list and must be
    # rejected loudly with TypeError -- NOT silently coerced into "0
    # requirements, perfect score" via the current `if not required_fields`
    # truthiness check. A perfect completeness score for a row that never
    # actually specified requirements is a false-positive PASS in an eval
    # report, which is worse than a crash: nobody investigates a green row.
    with pytest.raises(TypeError):
        calculate_completeness({"vendor": "Acme"}, None)  # type: ignore[arg-type]


def test_calculate_completeness_iterates_characters_when_given_a_bare_string() -> None:
    # Risk: `required_fields` is typed as `list[str]`, but the only guard
    # is `if not required_fields`. A non-empty string is truthy, so a
    # dataset bug that hands a single field name as a bare string instead
    # of a one-element list (`"vendor"` instead of `["vendor"]`) doesn't
    # raise -- it silently iterates *characters* of the string as if each
    # were its own required field path.
    found, total, score = calculate_completeness({"v": "x"}, "vendor")

    # "vendor" has 6 characters; only "v" happens to exist as a top-level
    # key in `data`, so exactly one of six bogus "fields" is found.
    assert total == 6
    assert found == 1


def test_calculate_field_accuracy_raises_for_non_dict_expected_data() -> None:
    # Risk: `expected_data` is typed `dict[str, Any]`, but only a
    # truthiness check guards it (`if not expected_data`). A non-empty
    # list -- plausible if an eval dataset's "expected" column was
    # accidentally loaded as a list of [key, value] pairs instead of a
    # dict -- passes the truthiness check and then crashes on
    # `.items()`, which lists don't have.
    with pytest.raises(AttributeError):
        calculate_field_accuracy(
            {"vendor": "Acme"},
            [("vendor", "Acme")],  # type: ignore[arg-type]
        )


def test_calculate_field_accuracy_crashes_on_non_string_dict_keys() -> None:
    # Risk: `field_name` from `expected_data.items()` is fed straight into
    # `get_nested_value`, which calls `.split(".")` on it. JSON object keys
    # are always strings, but a hand-built or programmatically merged
    # Python dict (e.g. from a pandas groupby) can easily have int keys.
    # This crashes with AttributeError instead of a clear "expected string
    # keys" validation error.
    with pytest.raises(AttributeError):
        calculate_field_accuracy({"vendor": "Acme"}, {1: "Acme"})  # type: ignore[dict-item]


def test_get_nested_value_returns_none_when_data_root_is_not_a_dict() -> None:
    # Risk: `get_nested_value` assumes `data` is a dict but never validates
    # it up front -- only the isinstance check deep inside the loop guards
    # it. Confirms non-dict roots (str, list, int, None) degrade to a
    # clean None instead of raising, for every plausible "wrong shape"
    # root a corrupted extraction result could produce.
    assert get_nested_value("not-a-dict", "vendor") is None
    assert get_nested_value(["not", "a", "dict"], "vendor") is None
    assert get_nested_value(42, "vendor") is None
    assert get_nested_value(None, "vendor") is None


# ---------------------------------------------------------------------------
# State & Side Effects
# ---------------------------------------------------------------------------


def test_evaluate_result_does_not_mutate_input_collections() -> None:
    # Risk: if evaluate_result (or anything it calls) ever mutated
    # `expected_data` or `required_fields` in place -- e.g. via a future
    # "pop matched fields as we go" optimization -- a caller reusing the
    # same expected-data dict across multiple eval cases (a very natural
    # eval-harness pattern) would see later cases silently scored against
    # a corrupted expectation. Pin down that inputs are read-only today.
    expected_data = {"vendor": "Acme", "total": 100.0}
    required_fields = ["vendor", "total"]
    expected_data_copy = dict(expected_data)
    required_fields_copy = list(required_fields)

    result = DocumentExtractionResult(
        document_type="invoice",
        data={"invoice_number": "INV-1", "vendor": "Acme", "total": 100.0},
        warnings=[],
    )

    evaluate_result(
        case_id="c1",
        expected_type="invoice",
        expected_data=expected_data,
        required_fields=required_fields,
        result=result,
    )

    assert expected_data == expected_data_copy
    assert required_fields == required_fields_copy


def test_duplicate_required_fields_inflate_completeness_denominator_and_numerator() -> None:
    # Risk: calculate_completeness treats required_fields as a flat
    # sequence, not a set, and never deduplicates. If an eval dataset
    # accidentally lists the same required field twice (e.g. a bad CSV
    # merge), that field is checked -- and counted -- twice. Total and
    # found both inflate identically, so the *score* happens to stay
    # correct, but `required_fields_total`/`required_fields_found` in any
    # downstream report ("7 of 7 fields found!") becomes misleading about
    # how many distinct fields actually exist.
    data = {"vendor": "Acme"}
    required_fields = ["vendor", "vendor", "vendor"]

    found, total, score = calculate_completeness(data, required_fields)

    assert total == 3  # only 1 distinct field, reported as 3
    assert found == 3
    assert score == 1.0


def test_evaluate_result_with_bypassed_validation_yields_silent_zero_score() -> None:
    # Risk: `_coerce_mapping` only special-cases `BaseModel`; anything else
    # is passed through untouched and assumed to already be a plain dict.
    # In normal operation pydantic guarantees `result.data` is always a
    # DocumentData subclass, but `model_construct` (used by tests, cached
    # fixtures, or any code path that bypasses `__init__` validation) can
    # produce a `DocumentExtractionResult` whose `.data` is an arbitrary
    # object. The scorer doesn't detect this corrupted state -- it just
    # silently reports 0% completeness/accuracy, which in a real eval
    # report reads identically to "the model extracted nothing," masking
    # what is actually a data-plumbing bug.
    corrupted_result = DocumentExtractionResult.model_construct(
        document_type="invoice",
        data="not-actually-a-document",  # bypasses the DocumentData union
        warnings=[],
    )

    evaluation = evaluate_result(
        case_id="corrupted",
        expected_type="invoice",
        expected_data={"vendor": "Acme"},
        required_fields=["vendor"],
        result=corrupted_result,
    )

    assert evaluation.completeness_score == 0.0
    assert evaluation.field_accuracy == 0.0


# ---------------------------------------------------------------------------
# Error Handling & Resilience
# ---------------------------------------------------------------------------


class _HostileEquality:
    """Simulates a poisoned object landing in eval data (e.g. a
    deserialization bug that leaves a non-primitive in the JSON tree)."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("comparison intentionally sabotaged")

    def __hash__(self) -> int:
        return 0


def test_values_match_propagates_exception_from_hostile_eq_object() -> None:
    # Risk: `values_match`'s final fallback is a bare `actual == expected`
    # with no try/except, unlike the float-tolerance branch which does
    # guard its conversions. Any object with a misbehaving `__eq__` --
    # plausible if a dataset loader ever leaves a raw exception, a
    # dataclass instance, or another non-JSON-primitive in the parsed
    # tree -- crashes the entire evaluation run instead of scoring that
    # one field as a mismatch.
    with pytest.raises(RuntimeError):
        values_match(_HostileEquality(), "expected string")


def test_get_nested_value_empty_path_must_raise_valueerror() -> None:
    # Contract under test: `path` is documented (via required_fields usage)
    # as a dotted/bracketed field locator, never "the whole document." An
    # empty string is not a valid locator and must raise ValueError -- NOT
    # silently fall through the regex with zero token matches and return
    # `data` completely untouched. Today, a required-field entry of `""`
    # (e.g. a trailing-comma CSV artifact in the eval dataset) is scored as
    # "found" for ANY non-empty document, regardless of what the field was
    # supposed to be -- a false-positive completeness match with no error
    # trail pointing back to the malformed dataset row.
    data = {"vendor": "Acme", "total": 100.0}

    with pytest.raises(ValueError):
        get_nested_value(data, "")


def test_get_nested_value_malformed_bracket_syntax_must_raise_valueerror() -> None:
    # Contract under test: the path-segment regex `([^[.\]]+)|\[(\d+)\]`
    # requires digits inside brackets. A non-numeric bracket expression
    # (a typo'd dataset path like "items[abc]" instead of "items[0]") is a
    # syntax error and must raise ValueError -- NOT be silently
    # reinterpreted as a second, unrelated dict-key lookup ("items.abc").
    # Today's silent reinterpretation means an authoring typo in the eval
    # dataset produces a confusing-but-successful traversal instead of a
    # stack trace pointing at the bad path string.
    data = {"items": {"abc": "unexpected match"}}

    with pytest.raises(ValueError):
        get_nested_value(data, "items[abc]")
