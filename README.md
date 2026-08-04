# pdf-data-extractor

[![Tests](https://github.com/E-techgod/pdf-data-extractor/actions/workflows/test.yml/badge.svg)](https://github.com/E-techgod/pdf-data-extractor/actions/workflows/test.yml)

A Groq-powered document extraction agent for text-based PDFs. It extracts PDF text, classifies the document as `invoice`, `resume`, `receipt`, `report`, or `generic`, then routes the result through a schema-validated extraction tool for that document type.

## How it works

The pipeline runs in four stages, all orchestrated from `classify_pdf_with_groq()` in `src/pdf_data_extractor/agent.py`:

1. **Load** — `extract_pdf_text()` reads text from each PDF page and rejects missing files, directories, non-PDF inputs, and PDFs with no extractable text.
2. **Classify** — `classify_with_groq()` forces Groq to call the `classify_document` tool, which validates the model's guess through Pydantic. That result is then **overridden** by a deterministic keyword scorer (`_classify_document_by_keywords()`), which always decides the final `document_type` (`invoice`, `resume`, `receipt`, `report`, or `generic`), falling back to `generic` when keyword signal is weak or tied between categories. The Groq classification call still has to succeed, but its output is discarded — the keyword scorer is the actual source of truth.
3. **Route** — `tool_registry.py` maps the final `document_type` to a specialized extraction tool schema. If no tool is registered for the type, extraction is skipped and an empty result with a warning is returned instead.
4. **Extract** — Groq is forced to call the matching `extract_*_fields` tool with a document-type-specific prompt hint. The tool implementation in `tools.py` builds and validates the output through the Pydantic models in `schemas.py`, returning a `DocumentExtractionResult`.

The same `classify_with_groq()` core (exposed as `analyze_document_with_groq()`) is reused by the evaluation runner in `evals/run_evals.py`, so classification and extraction are always exercised through one code path whether the input is a PDF or raw text.

## Project layout

```text
src/pdf_data_extractor/
  agent.py          # Groq orchestration, keyword scoring, tool-call execution
  config.py         # loads GROQ_API_KEY from .env
  evaluation.py     # completeness and field-accuracy scoring helpers
  main.py           # package CLI entrypoint used by `pdf-extractor`
  pdf_loader.py     # PDF text extraction
  schemas.py        # Pydantic models for classification and extraction payloads
  tool_registry.py  # document type -> extraction tool mapping
  tool_schemas.py   # Groq tool schemas
  tools.py          # schema-validated tool implementations
evals/
  golden_dataset.json  # labeled evaluation cases
  run_evals.py         # eval runner for text cases
tests/
  test_agent.py
  test_chaos_agent.py
  test_evaluations.py
  test_chaos_evaluations.py
  test_pdf_loader.py
  test_chaos_pdf_loader.py
  test_tools.py
  test_chaos_tools.py
  test_main.py
graphify-out/
  GRAPH_REPORT.md   # graph summary report
  graph.json        # graph data
  graph.html        # interactive graph view
```

## Architecture

```mermaid
flowchart TD
    U["CLI: pdf-extractor data/file.pdf"] --> M["main() — src/pdf_data_extractor/main.py"]
    M --> CPG[classify_pdf_with_groq]
    CPG --> EXT[extract_pdf_text]
    EXT --> PDF[(PDF pages)]
    CPG --> CWG[classify_with_groq]

    CWG --> G1["Groq: forced classify_document call"]
    G1 --> T1[tools.classify_document]
    T1 --> LLMC["model classification\n(validated, then discarded)"]

    CWG --> KW["_classify_document_by_keywords()"]
    KW --> FINAL["final document_type\n(keyword scorer always wins)"]

    FINAL --> LOOKUP{"tool_registry.py\nSPECIALIZED_TOOL_NAMES"}
    LOOKUP -->|"no registered tool"| EMPTY["EmptyExtractionData\n+ warning"]
    LOOKUP -->|"tool registered"| G2["Groq: forced extract_*_fields call"]

    G2 --> T2["tools.extract_*_fields"]
    T2 --> SCH["schemas.py Pydantic models"]
    SCH --> RESULT[DocumentExtractionResult]
    EMPTY --> RESULT
    RESULT --> M

    RE[evals/run_evals.py] --> GD["golden_dataset.json"]
    RE --> ADG[analyze_document_with_groq]
    ADG --> CWG
    RE --> EVAL["evaluation.py: evaluate_result"]
    EVAL --> RESULT
```

## Running it

```bash
uv sync
```

Create a `.env` file at the repo root with:

```bash
GROQ_API_KEY=your_key_here
# optional override
GROQ_MODEL=llama-3.1-8b-instant
```

Run the root script against the default sample:

```bash
uv run main.py
```

Run it against a specific PDF:

```bash
uv run main.py data/receipt.pdf
```

Or use the packaged CLI entrypoint:

```bash
pdf-extractor data/receipt.pdf
```

## Evaluation

The repo now includes a dataset-driven evaluation path in `evals/`. `run_evals.py` feeds labeled text cases through `analyze_document_with_groq()` and scores:

- classification correctness
- required-field completeness
- exact checked-field accuracy

Run it with:

```bash
uv run python evals/run_evals.py
```

## Tests

```bash
uv run pytest
```

Current collected suite: `125` tests.

The test coverage now spans:

- orchestration and tool-call enforcement
- schema validation for all extraction types
- PDF loading behavior
- evaluation metrics and nested-field matching
- adversarial and chaos cases for agent, tools, loader, and evaluation logic

## Graph artifacts

The generated architecture graph lives in `graphify-out/`:

- `GRAPH_REPORT.md` summarizes communities, hubs, and graph freshness
- `graph.json` stores the underlying graph data
- `graph.html` renders an interactive local network view

The current graph report dated `2026-08-03` describes a graph with `231` nodes, `625` edges, and `24` communities. It highlights `classify_with_groq()`, `DocumentExtractionResult`, `extract_pdf_text()`, and the evaluation helpers as central nodes in the codebase.

## Status

The current version supports:

- PDF text extraction from text-based PDFs
- deterministic keyword-assisted classification with `generic` fallback
- Groq-driven extraction for `invoice`, `resume`, `receipt`, `report`, and `generic`
- Pydantic validation for every extraction payload
- dataset-based evaluation utilities
- graph documentation outputs under `graphify-out/`
