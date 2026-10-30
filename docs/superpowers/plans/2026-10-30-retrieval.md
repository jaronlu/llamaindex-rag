# Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local vector-index retrieval stage that returns ranked, complete, source-traceable PDF chunks with explicit model compatibility checks.

**Architecture:** `src/rag/retrieval/search.py` loads `manifest.json`, verifies the requested Hugging Face snapshot identity against the saved index configuration, loads the persisted `VectorStoreIndex`, and returns immutable result records containing rank, similarity score, node ID, text, and source metadata. `src/rag/cli/retrieval.py` parses a question and Top-K, invokes the core function, and prints readable results plus explicit no-result/source warnings. Existing indexing storage and metadata remain the source of truth.

**Tech Stack:** Python 3.12, LlamaIndex `VectorStoreIndex`/`StorageContext`, local `HuggingFaceEmbedding`, unittest, argparse.

**Spec:** `docs/prd/PRD.md` section 6.4 and section 7.

## Global Constraints

- Every result must show complete text, source file, physical PDF page(s), node ID, rank, and retrieval score.
- PDF physical pages start at 1; cross-page nodes show the complete page range from persisted metadata.
- Query model and persisted index configuration must match; incompatible configurations fail clearly.
- Missing index, missing model, malformed metadata, missing source metadata, and no results must be readable failures; no result with incomplete provenance is silently delivered as traceable.
- Support multiple documents through persisted metadata; do not hardcode the sample PDF or questions.
- Do not send document or query text to an external service; use a local snapshot and never hardcode credentials.

## Review Focus

- Querying an index with a same-named but changed model snapshot must be rejected by the saved model digest.
- A node with absent or invalid source/page metadata must be marked untraceable and excluded from delivered results.
- Top-K values of zero, negative, and larger than available nodes must be handled explicitly.
- A persisted index can be reloaded without reparsing the source PDF.
- A question with no useful candidates must print an explicit no-result message.

### Task 1: Retrieval core

**Files:**
- Create: `src/rag/retrieval/__init__.py`
- Create: `src/rag/retrieval/search.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: an existing index directory, a local HF snapshot path, a question, and `top_k: int`.
- Produces: `RetrievalResult` records and `retrieve(index_path, model_path, question, top_k=5) -> list[RetrievalResult]`.

- [ ] Write failing tests for ranked results with score/node/source fields, compatibility rejection, provenance exclusion, Top-K validation, and reload without source PDF.
- [ ] Run `uv run python -m unittest tests.test_retrieval -v` and verify the feature-missing failures.
- [ ] Implement the minimum core using LlamaIndex persisted storage and the existing `snapshot_identity`/`check_compatibility` helpers.
- [ ] Re-run the focused tests and then the full suite.

### Task 2: Retrieval CLI

**Files:**
- Create: `src/rag/cli/retrieval.py`
- Modify: `src/rag/cli/__init__.py` only if an export is needed
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 `retrieve` function.
- Produces: `main(argv: list[str] | None = None) -> int` with `search INDEX QUESTION --model-path PATH [--top-k N]` behavior.

- [ ] Write failing CLI tests for readable complete output, no-result output, and clear errors.
- [ ] Run the focused CLI tests and verify failure.
- [ ] Implement the command with the project logging decorator and source-aware output.
- [ ] Document the command, score ordering, model requirements, and a reproducible validation workflow.
- [ ] Run the full test suite and type checks.

### Task 3: Real-index validation record

**Files:**
- Create or modify: `docs/experiments/retrieval.md`
- Modify: `README.md` if command details need a final adjustment

- [ ] Run retrieval against the user-confirmed sample index with three in-document questions and one out-of-document question.
- [ ] Record model, index, Top-K, returned sources, manual judgments, and limitations without claiming unperformed checks.
- [ ] Run final verification commands and review the complete diff.
