"""用固定问题集统计答案页在 dense 和重排结果中的排名。

本模块复用 ``search.retrieve`` 和 ``search.rerank``，只读取已保存的索引，不重新解析
PDF，也不调用外部服务。它回答的是“人工标注的答案页有没有被召回、排在第几”，仍需要
回到 PDF 核对片段是否真的支持答案；自动排名不能替代人工判断。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from .search import RetrievalResult, rerank, retrieve


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """带人工标注的问题。``answer_pages`` 为空表示文档内没有明确答案。"""

    question_id: str
    question: str
    answer_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuestionEvaluation:
    """一个问题在两种检索模式下的答案页排名。

    ``dense_rank`` 和 ``rerank_rank`` 都只在交付的前 ``top_k`` 条里计算，便于直接
    比较；``in_candidates`` 表示答案页是否进入 dense 的 ``candidate_k`` 候选集，
    用来区分“没召回到”和“召回到了但排不上去”这两种完全不同的失败原因。
    """

    question: EvaluationQuestion
    dense_rank: int | None
    rerank_rank: int | None
    in_candidates: bool
    dense_pages: tuple[tuple[int, ...], ...]
    rerank_pages: tuple[tuple[int, ...], ...]
    top_rerank_score: float | None


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """读取问题集 JSON 并校验字段。

    JSON 是对象数组，每项包含 ``id``、``question`` 和 ``answer_pages``（正整数数组，
    文档外问题用空数组）。字段缺失或类型不对时直接报错，避免标注写错后被当成
    “没有召回到”。
    """
    try:
        raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"问题集无法读取: {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("问题集必须是非空数组")

    questions: list[EvaluationQuestion] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"问题集第 {index} 项必须是对象")
        question_id = item.get("id")
        question = item.get("question")
        pages = item.get("answer_pages")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"问题集第 {index} 项缺少 id")
        if question_id in seen:
            raise ValueError(f"问题集 id 重复: {question_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"问题集 {question_id} 缺少 question")
        if not isinstance(pages, list) or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages):
            raise ValueError(f"问题集 {question_id} 的 answer_pages 必须是正整数数组")
        seen.add(question_id)
        questions.append(EvaluationQuestion(question_id, question.strip(), tuple(pages)))
    return questions


def answer_rank(results: Sequence[RetrievalResult], answer_pages: Sequence[int]) -> int | None:
    """返回答案页第一次出现的排名；没有命中返回 ``None``。

    片段可能跨页，所以比较的是页码集合是否相交，而不是只看首页。
    """
    if not answer_pages:
        return None
    pages = set(answer_pages)
    for result in results:
        if pages.intersection(result.page_numbers):
            return result.rank
    return None


def evaluate_questions(
    index_path: Path,
    model_path: Path,
    rerank_model_path: Path,
    questions: Sequence[EvaluationQuestion],
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[QuestionEvaluation]:
    """对每个问题运行 dense 和重排，记录答案页排名。

    每个问题都重新加载索引和模型，问题多时耗时会线性增长；这样换来的是每条结果都
    走与单独运行 ``search`` 完全相同的流程，避免评估路径和实际检索分叉。
    """
    evaluations: list[QuestionEvaluation] = []
    for question in questions:
        candidates = retrieve(index_path, model_path, question.question, top_k=candidate_k)
        reranked = rerank(question.question, candidates, rerank_model_path, top_k=top_k)
        evaluations.append(
            QuestionEvaluation(
                question=question,
                dense_rank=answer_rank(candidates[:top_k], question.answer_pages),
                rerank_rank=answer_rank(reranked, question.answer_pages),
                in_candidates=answer_rank(candidates, question.answer_pages) is not None,
                dense_pages=tuple(result.page_numbers for result in candidates[:top_k]),
                rerank_pages=tuple(result.page_numbers for result in reranked),
                top_rerank_score=reranked[0].rerank_score if reranked else None,
            )
        )
    return evaluations


def format_pages(pages: Sequence[int]) -> str:
    """把页码元组格式化成 ``291`` 或 ``291–292``，用于命令行展示。"""
    if not pages:
        return "无"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}–{pages[-1]}"
