# Dense Recall and Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local two-stage retrieval path that recalls up to 20 dense candidates and reranks them locally down to 5 traceable results.

**Architecture:** Keep `retrieve()` as the dense baseline. Add a separate `rerank()` function using a local sentence-transformers CrossEncoder and a `retrieve_and_rerank()` orchestration function that calls dense retrieval with `candidate_k=20`, then returns `top_k=5` results with both original and rerank metadata. Extend the existing CLI with optional reranker-model and candidate-count arguments while preserving dense-only mode.

**Tech Stack:** Python 3.12, LlamaIndex persisted `VectorStoreIndex`, local `sentence-transformers.CrossEncoder`, unittest.

**Spec:** `docs/prd/PRD.md` sections 6.5 and 8.

## Global Constraints

- Reranking must use a local model path and `local_files_only=True`; document and query text must not be sent to external services.
- Dense retrieval remains the source of candidates and must keep its model compatibility and provenance checks.
- Final results must preserve complete text, file, physical page(s), node ID, dense score, original rank, rerank score, and new rank.
- Rerank scores and dense scores are separate values and must not be compared as if they share a scale.
- Missing or invalid inputs must produce readable errors; no silent fallback from rerank mode to dense-only mode.
- Existing dense-only callers remain usable for baseline comparisons.

## Review Focus

- A relevant candidate moves upward while its dense score and original rank remain available.
- Fewer than 20 available candidates are handled without fabricating results.
- `top_k` and `candidate_k` zero, negative, boolean, and inconsistent values are rejected.
- A missing reranker model path fails before model loading.
- CLI output distinguishes dense-only results from reranked results and shows both scores.

### Task 1: Reranking core

**Files:**
- Modify: `src/rag/retrieval/search.py`
- Modify: `src/rag/retrieval/__init__.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Existing: `retrieve(index_path, model_path, question, top_k=5) -> list[RetrievalResult]`.
- Add: `rerank(question: str, candidates: Sequence[RetrievalResult], model_path: Path, top_k: int = 5) -> list[RetrievalResult]`.
- Add: `retrieve_and_rerank(index_path: Path, model_path: Path, rerank_model_path: Path, question: str, candidate_k: int = 20, top_k: int = 5) -> list[RetrievalResult]`.
- Extend `RetrievalResult` with optional `original_rank` and `rerank_score` fields; dense results keep them unset.

- [x] Write failing tests for reorder metadata, orchestration with candidate 20/final 5, validation, and local model-path errors.
- [x] Run `uv run python -m unittest tests.test_retrieval -v` and confirm the new tests fail because the interfaces are missing.
- [x] Implement local CrossEncoder scoring, deterministic descending rerank order, and orchestration.
- [x] Re-run focused tests, then the full test suite and mypy.

### Task 2: CLI and documentation

**Files:**
- Modify: `src/rag/cli/retrieval.py`
- Modify: `tests/test_retrieval.py`
- Modify: `README.md`
- Create: `docs/experiments/reranking.md`

**Interfaces:**
- Add optional `--rerank-model-path PATH` and `--candidate-k N` to `search`; when reranker path is supplied, run dense candidate retrieval then rerank, otherwise keep dense baseline.
- Default rerank-mode values are candidate 20 and final top-k 5.
- Reranked output shows original rank, new rank, dense score, and rerank score.

- [x] Write failing CLI tests for reranked output and clear reranker errors.
- [x] Run the focused CLI tests and confirm failure.
- [x] Implement CLI branching without silently falling back when rerank mode is selected.
- [x] Document local model requirements and the reproducible `dense top 20 → rerank top 5` command.
- [x] Record the experiment protocol and pending manual validation without claiming quality improvement.
- [x] Run the full test suite, type checks, and `git diff --check`.


