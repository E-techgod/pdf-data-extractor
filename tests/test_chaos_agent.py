"""
Chaos / adversarial-input tests for src/pdf_data_extractor/agent.py.

These target the Groq-facing orchestration layer: malformed API responses,
argument payloads that are technically parseable but semantically hostile,
and defensive code paths that exist but are never actually exercised by the
real control flow. Every test comment states the concrete failure scenario
it guards against or documents.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.pdf_data_extractor.agent import (
    _execute_tool_call,
    _parse_tool_arguments,
    classify_with_groq,
)
from src.pdf_data_extractor.schemas import EmptyExtractionData
from src.pdf_data_extractor.tool_registry import SPECIALIZED_TOOL_NAMES


def make_tool_call(
    *,
    name: str,
    arguments: str | dict,
    call_id: str = "call_123",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_response(
    *,
    tool_calls: list[SimpleNamespace] | None = None,
    content: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def make_fake_client(*responses: SimpleNamespace) -> Mock:
    client = Mock()
    client.chat.completions.create.side_effect = list(responses)
    return client


def test_empty_choices_list_crashes_with_unhandled_indexerror() -> None:
    # Risk: _request_required_tool_call does
    #     assistant_message = response.choices[0].message
    # with no length check. The Groq/OpenAI-style API contract does not
    # guarantee a non-empty `choices` array (e.g. a response fully blocked
    # by content filtering can legitimately return `choices: []`). A few
    # lines later the code DOES defensively check
    # `if not tool_calls: raise RuntimeError(...)` -- proving the author
    # anticipated "the model didn't produce what we need" as a real
    # scenario -- but that defensive pattern was never applied to
    # `choices` itself. The result is a raw, uncontextualized IndexError
    # instead of a clean, catchable RuntimeError.
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])

    with pytest.raises(IndexError):
        classify_with_groq("Some document text", client=client)


def test_execute_tool_call_does_not_convert_attributeerror_to_valueerror() -> None:
    # Risk: _execute_tool_call's error handling is
    #     except TypeError as exc:
    #         raise ValueError(...)
    # This only converts *signature-mismatch* failures (wrong/missing
    # kwargs) into a clean, documented ValueError. It does nothing for
    # exceptions raised *inside* a tool function body once the call
    # succeeds argument-wise -- such as extract_invoice_fields'
    # `currency.upper()` blowing up on `currency=None`. A syntactically
    # valid tool call (`{"currency": null}` is valid JSON, `null` is a
    # legal value for an optional field per the tool schema) still crashes
    # the whole pipeline with an AttributeError the caller has no
    # documented reason to expect or catch.
    tool_call = make_tool_call(
        name="extract_invoice_fields",
        arguments='{"vendor": "Example Services LLC", "currency": null}',
    )

    with pytest.raises(AttributeError):
        _execute_tool_call(tool_call)


def test_duplicate_json_keys_silently_resolve_to_last_value() -> None:
    # Risk: _parse_tool_arguments hands the raw string straight to
    # json.loads, which per the JSON spec is free to (and in CPython does)
    # silently keep only the LAST occurrence of a duplicate key, discarding
    # earlier ones without any error or warning. If a malformed/adversarial
    # tool-call payload contains a duplicate "document_type" key -- whether
    # from a buggy model, a prompt-injection attempt, or an upstream
    # transport bug -- the classification silently becomes whichever value
    # came last, with zero signal that a collision occurred.
    parsed = _parse_tool_arguments(
        '{"document_type": "invoice", "document_type": "resume", "reason": "x"}',
        tool_name="classify_document",
    )

    assert parsed == {"document_type": "resume", "reason": "x"}


def test_ast_literal_eval_fallback_silently_accepts_non_json_python_syntax() -> None:
    # Risk: when json.loads fails, _parse_tool_arguments falls back to
    # ast.literal_eval. This is safe from code execution (literal_eval
    # cannot run arbitrary code), but it means a string that is NOT valid
    # JSON -- e.g. single-quoted Python dict syntax that a model or a
    # buggy client might emit instead of proper JSON -- is silently
    # accepted rather than rejected as malformed. Callers relying on
    # "valid JSON in, or a clean error" get a wider, undocumented
    # acceptance surface instead.
    parsed = _parse_tool_arguments(
        "{'document_type': 'invoice', 'reason': 'ok'}",
        tool_name="classify_document",
    )

    assert parsed == {"document_type": "invoice", "reason": "ok"}


def test_unregistered_document_type_fallback_is_unreachable_in_real_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Risk (dead defensive code / schema-registry drift): classify_with_groq
    # falls back to `_empty_result_for_document_type` whenever
    # `SPECIALIZED_TOOL_NAMES.get(document_type)` returns None. But
    # `document_type` only ever reaches this point after surviving
    # Pydantic validation against the `DocumentType` Literal in
    # schemas.py, whose 5 values are IDENTICAL to
    # SPECIALIZED_TOOL_NAMES' 5 keys. That means through the real,
    # LLM-driven code path, this fallback branch can NEVER execute --
    # every valid classification is guaranteed a registered extractor.
    # It is a "safety net" that has never actually caught anything.
    # This test proves that fact (sanity check) and then forces the
    # branch by simulating registry/schema drift (e.g. someone adds a
    # 6th DocumentType literal without registering its extractor), to
    # confirm the fallback still behaves correctly *when* it is finally
    # exercised -- because nothing else in the suite ever exercises it
    # through classify_with_groq itself.
    assert set(SPECIALIZED_TOOL_NAMES) == {
        "invoice",
        "resume",
        "receipt",
        "report",
        "generic",
    }

    monkeypatch.delitem(SPECIALIZED_TOOL_NAMES, "invoice")

    classify_call = make_tool_call(
        name="classify_document",
        arguments=('{"document_type": "invoice", "reason": "Drifted registry."}'),
    )
    client = make_fake_client(make_response(tool_calls=[classify_call]))

    result = classify_with_groq(
        "Invoice Number: 123. Amount Due: $50.",
        client=client,
    )

    assert result.document_type == "invoice"
    assert isinstance(result.data, EmptyExtractionData)
    assert result.warnings == [
        "No specialized extractor is registered for document_type 'invoice'."
    ]
    # Only one Groq call happened -- classification -- because the
    # (now-missing) extractor route was never invoked.
    assert client.chat.completions.create.call_count == 1
