"""检索命令行入口，展示候选片段、相似度和可核对来源。

除单次检索外，还提供 ``evaluate``：用带答案页标注的固定问题集，统计答案页在
dense 和重排结果中的排名，供比较两阶段检索的效果。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .logging import logged_command

from ..config.common import DEFAULT_EMBEDDING_MODEL_PATH, DEFAULT_RERANK_MODEL_PATH
from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from ..retrieval.evaluate import evaluate_questions, load_questions
from ..retrieval.search import format_pages, retrieve, retrieve_and_rerank


logger = logging.getLogger(__name__)


def _run_evaluation(args: argparse.Namespace) -> int:
    """运行问题集评估，打印每个问题的答案页排名和汇总。"""
    try:
        questions = load_questions(args.questions)
        evaluations = evaluate_questions(
            args.index,
            args.model_path,
            args.rerank_model_path,
            questions,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
    except Exception as exc:
        logger.exception("retrieval.evaluate_failed index=%s questions=%s", args.index, args.questions)
        print(f"评估失败: {exc}", file=sys.stderr)
        return 1

    in_document = [item for item in evaluations if item.question.answer_pages]
    outside = [item for item in evaluations if not item.question.answer_pages]
    print(f"问题集: {args.questions.expanduser().resolve()}（{len(evaluations)} 个问题，其中文档外 {len(outside)} 个）")
    print(f"索引: {args.index.expanduser().resolve()}")
    print(f"检索模式: dense top {args.candidate_k} → rerank top {args.top_k}")
    print(f"说明: dense/rerank 名次只在交付的前 {args.top_k} 条内计算；候选内表示答案页是否进入 dense 的 {args.candidate_k} 条候选。名次命中仍须人工回到 PDF 核对原文。")

    for item in evaluations:
        dense = "—" if item.dense_rank is None else f"第 {item.dense_rank}"
        reranked = "—" if item.rerank_rank is None else f"第 {item.rerank_rank}"
        candidates = "—" if not item.question.answer_pages else ("是" if item.in_candidates else "否")
        score = "—" if item.top_rerank_score is None else f"{item.top_rerank_score:.4f}"
        print(f"\n{item.question.question_id} | {item.question.question}")
        print(f"  答案页: {format_pages(item.question.answer_pages)} | dense: {dense} | rerank: {reranked} | 候选内: {candidates} | 最高重排分: {score}")
        print(f"  dense 页码: {[format_pages(pages) for pages in item.dense_pages]}")
        print(f"  rerank 页码: {[format_pages(pages) for pages in item.rerank_pages]}")

    if in_document:
        dense_hits = sum(1 for item in in_document if item.dense_rank is not None)
        rerank_hits = sum(1 for item in in_document if item.rerank_rank is not None)
        recalled = sum(1 for item in in_document if item.in_candidates)
        print(f"\n汇总（文档内 {len(in_document)} 题）: 答案页进入候选 {recalled}/{len(in_document)}；dense 前 {args.top_k} 命中 {dense_hits}/{len(in_document)}；rerank 前 {args.top_k} 命中 {rerank_hits}/{len(in_document)}")
    if outside:
        scores = [item.top_rerank_score for item in outside if item.top_rerank_score is not None]
        if scores:
            print(f"书外问题 {len(outside)} 题，最高重排分 {max(scores):.4f}；该分数只能与同一模型下的文档内问题对照，程序不会据此判定资料不足。")
    return 0


@logged_command
def main(argv: list[str] | None = None) -> int:
    """从已保存索引检索问题并输出完整候选片段。

    命令不重新读取 PDF；正文和页码来自持久化 Node。源文件消失时仍展示索引中保存
    的正文，但会明确标记为无法回到当前文件核对，避免把缺失来源静默当作已验证。
    """
    parser = argparse.ArgumentParser(description="从持久化向量索引检索并显示可追溯片段")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="检索一个自然语言问题")
    search.add_argument("index", type=Path, help="索引目录")
    search.add_argument("question", help="自然语言问题")
    search.add_argument("--model-path", type=Path, default=DEFAULT_EMBEDDING_MODEL_PATH, help="覆盖默认的 Hugging Face embedding 模型快照目录")
    search.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help=f"最多显示的候选数量（默认 {DEFAULT_RETRIEVAL_TOP_K}）")
    search.add_argument("--rerank-model-path", type=Path, default=DEFAULT_RERANK_MODEL_PATH, help="覆盖配置文件中的本地 CrossEncoder 模型目录")
    search.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K, help=f"重排前 dense 粗召回数量（默认 {DEFAULT_CANDIDATE_K}）")
    search.add_argument("--dense-only", action="store_true", help="仅运行 dense 基线，不使用 CrossEncoder 重排")
    evaluate = subparsers.add_parser("evaluate", help="用固定问题集统计答案页排名")
    evaluate.add_argument("index", type=Path, help="索引目录")
    evaluate.add_argument("questions", type=Path, help="问题集 JSON 路径，需包含人工标注的答案页")
    evaluate.add_argument("--model-path", type=Path, default=DEFAULT_EMBEDDING_MODEL_PATH, help="覆盖默认的 Hugging Face embedding 模型快照目录")
    evaluate.add_argument("--rerank-model-path", type=Path, default=DEFAULT_RERANK_MODEL_PATH, help="覆盖配置文件中的本地 CrossEncoder 模型目录")
    evaluate.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K, help=f"重排前 dense 粗召回数量（默认 {DEFAULT_CANDIDATE_K}）")
    evaluate.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help=f"计算名次时使用的交付数量（默认 {DEFAULT_RETRIEVAL_TOP_K}）")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return _run_evaluation(args)

    try:
        rerank_enabled = not args.dense_only
        if rerank_enabled:
            results = retrieve_and_rerank(
                args.index,
                args.model_path,
                args.rerank_model_path,
                args.question,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
            )
        else:
            results = retrieve(args.index, args.model_path, args.question, args.top_k)
    except Exception as exc:
        logger.exception("retrieval.failed index=%s question_length=%s", args.index, len(args.question))
        print(f"检索失败: {exc}", file=sys.stderr)
        return 1

    print(f"问题: {args.question}")
    print(f"索引: {args.index.expanduser().resolve()}")
    print(f"Top-K: {args.top_k}")
    if not args.dense_only:
        print(f"检索模式: dense top {args.candidate_k} → rerank top {args.top_k}")
        print("分数说明: dense 相似度分数和 rerank 分数来自不同模型，不能直接比较。")
    else:
        print("检索模式: dense")
        print("分数说明: 相似度分数越高表示向量越相似；分数只在同一索引和查询配置下比较。")
    if not results:
        print("未找到可交付的检索结果：没有候选片段通过来源和正文检查。")
        return 0

    for result in results:
        score = "无" if result.score is None else f"{result.score:.6f}"
        print(f"\n--- 结果 {result.rank} ---")
        print(f"排名: {result.rank}")
        if result.rerank_score is not None:
            print(f"原始排名: {result.original_rank}")
            print(f"新排名: {result.rank}")
            print(f"dense 相似度分数: {score}")
            print(f"rerank 分数: {result.rerank_score:.6f}")
        else:
            print(f"相似度分数: {score}")
        print(f"片段 ID: {result.node_id}")
        print(f"文件名: {result.file_name}")
        print(f"来源路径: {result.source_path}")
        print(f"物理页码: {result.page_label}")
        if result.provenance_complete:
            print("来源状态: 可回到当前 PDF 核对")
        else:
            print(f"来源状态: 无法完成当前文件核对；{result.provenance_warning or '来源信息不完整'}")
        print("完整原文:")
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
