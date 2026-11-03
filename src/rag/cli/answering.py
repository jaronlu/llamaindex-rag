"""基于来源的回答命令，展示回答、引用和可核对的原文。

命令先复用检索模块取得片段，再把问题和片段发送给生成模型。密钥只从环境变量
读取；调用前会明确显示发送目标，避免把外部服务调用和本地检索混在一起。回答中
的引用编号会与交付片段逐一核对，越界或缺失都会提示，不能把生成成功当作回答正确。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .arguments import add_retrieval_arguments
from .logging import logged_command

from ..answering.generate import (
    Answer,
    create_llm,
    describe_llm_error,
    generate_answer,
    resolve_api_key,
    select_relevant_evidence,
)
from ..config.answering import (
    DEFAULT_ANSWER_API_BASE,
    DEFAULT_ANSWER_API_KEY_ENV,
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_ANSWER_TEMPERATURE,
    DEFAULT_ANSWER_TIMEOUT_SECONDS,
    DEFAULT_ANSWER_TRUST_ENV,
    DEFAULT_MIN_DENSE_SCORE,
    DEFAULT_MIN_RERANK_SCORE,
)
from ..config.retrieval import DEFAULT_RETRIEVAL_TOP_K
from ..retrieval.search import RetrievalResult, retrieve, retrieve_and_rerank


logger = logging.getLogger(__name__)


def _print_evidence(evidence: Sequence[RetrievalResult], cited: set[int]) -> None:
    """打印每条证据的完整正文和来源状态，并标出是否被回答引用。

    生成失败时也会用空引用集合调用，让已经检索到的片段仍可人工排查。
    """
    for number, result in enumerate(evidence, start=1):
        marker = "已被引用" if number in cited else "未被引用"
        print(f"\n--- 证据 {number}（{marker}）---")
        print(f"文件名: {result.file_name}")
        print(f"来源路径: {result.source_path}")
        print(f"物理页码: {result.page_label}")
        print(f"片段 ID: {result.node_id}")
        print(f"来源状态: {result.source_status_label}")
        print("完整原文:")
        print(result.text)


def _print_citation_check(answer: Answer) -> None:
    """输出引用编号和来源状态检查，明确区分可核对和不可核对的情况。"""
    if answer.invalid_citations:
        labels = ", ".join(f"[{number}]" for number in answer.invalid_citations)
        print(f"\n引用检查: 回答包含不存在的证据编号 {labels}；这些编号无法对应原文，不能作为依据。")
    elif answer.citations:
        labels = ", ".join(f"[{number}]" for number in answer.citations)
        print(f"\n引用检查: 回答引用了 {labels}。")
    else:
        print("\n引用检查: 回答没有出现 [n] 引用。若结论是资料不足，请人工确认是否确实缺乏依据；若是正常回答，则无法把陈述对应到原文。")

    # 被引用的证据仍可能因源文件缺失或版本变化而无法核对；不单独提示的话，
    # 读者容易把“有页码”误当成“已验证”。
    cited = set(answer.citations) - set(answer.invalid_citations)
    unverified = [number for number in sorted(cited) if not answer.evidence[number - 1].provenance_complete]
    if unverified:
        labels = ", ".join(f"[{number}]" for number in unverified)
        print(f"来源检查: 引用的证据 {labels} 无法回到当前文件核对；回答中的相应内容不能视为已验证。")


@logged_command
def main(argv: list[str] | None = None) -> int:
    """检索片段并生成带引用的回答。

    命令不重新解析 PDF；证据正文和页码来自持久化索引。没有可交付片段时直接给出
    资料不足提示，不调用生成模型，也不要求配置密钥。
    """
    parser = argparse.ArgumentParser(description="基于检索片段生成带引用的回答")
    parser.add_argument("index", type=Path, help="索引目录")
    parser.add_argument("question", help="自然语言问题")
    add_retrieval_arguments(
        parser,
        top_k_help=f"作为证据交付的片段数量（默认 {DEFAULT_RETRIEVAL_TOP_K}）",
        dense_only_help="仅使用 dense 检索结果作为证据，不重排",
    )
    parser.add_argument("--llm-model", default=DEFAULT_ANSWER_MODEL, help=f"生成模型名称（默认 {DEFAULT_ANSWER_MODEL}）")
    parser.add_argument("--api-base", default=DEFAULT_ANSWER_API_BASE, help="OpenAI 兼容服务地址")
    parser.add_argument("--api-key-env", default=DEFAULT_ANSWER_API_KEY_ENV, help=f"保存密钥的环境变量名（默认 {DEFAULT_ANSWER_API_KEY_ENV}）")
    parser.add_argument(
        "--no-llm-proxy",
        dest="llm_trust_env",
        action="store_false",
        default=DEFAULT_ANSWER_TRUST_ENV,
        help="生成请求不读取系统和环境变量代理，直连 API 地址",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=DEFAULT_MIN_RERANK_SCORE,
        help=f"重排证据的最低相关分（默认 {DEFAULT_MIN_RERANK_SCORE}；换模型或索引需重新校准）",
    )
    parser.add_argument(
        "--min-dense-score",
        type=float,
        default=DEFAULT_MIN_DENSE_SCORE,
        help=f"dense 证据的最低相似度（默认 {DEFAULT_MIN_DENSE_SCORE}；换模型或索引需重新校准）",
    )
    args = parser.parse_args(argv)

    try:
        if args.dense_only:
            results = retrieve(args.index, args.model_path, args.question, args.top_k)
        else:
            results = retrieve_and_rerank(
                args.index,
                args.model_path,
                args.rerank_model_path,
                args.question,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
            )
    except Exception as exc:
        logger.exception("answering.retrieval_failed index=%s question_length=%s", args.index, len(args.question))
        print(f"回答失败: {exc}", file=sys.stderr)
        return 1

    candidate_count = len(results)
    results = select_relevant_evidence(
        results,
        min_rerank_score=args.min_rerank_score,
        min_dense_score=args.min_dense_score,
    )

    print(f"问题: {args.question}")
    print(f"索引: {args.index.expanduser().resolve()}")
    print(f"Top-K: {args.top_k}")
    print("检索模式: dense" if args.dense_only else f"检索模式: dense top {args.candidate_k} → rerank top {args.top_k}")
    print(f"相关度门槛: dense ≥ {args.min_dense_score} / rerank ≥ {args.min_rerank_score}")
    if candidate_count != len(results):
        print(f"相关度过滤: {candidate_count} 条候选中 {candidate_count - len(results)} 条低于门槛，未作为证据。")
    print(f"生成模型: {args.llm_model}")
    print(f"API 地址: {args.api_base}")
    print(f"生成请求代理: {'跟随系统和环境变量' if args.llm_trust_env else '直连（--no-llm-proxy）'}")

    if not results:
        answer = generate_answer(args.question, results)
        print("\n资料不足: 没有候选片段通过来源、正文和相关度检查，未调用生成模型。")
        print(f"回答: {answer.answer}")
        return 0

    print(f"外部调用说明: 本次会把问题和 {len(results)} 条片段发送到上述地址；不会发送 PDF 文件、索引或其他文档。")
    try:
        api_key = resolve_api_key(args.api_key_env)
        llm = create_llm(
            model=args.llm_model,
            api_base=args.api_base,
            api_key=api_key,
            temperature=DEFAULT_ANSWER_TEMPERATURE,
            max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
            timeout=DEFAULT_ANSWER_TIMEOUT_SECONDS,
            trust_env=args.llm_trust_env,
        )
        answer = generate_answer(args.question, results, llm)
    except Exception as exc:
        logger.exception("answering.generation_failed model=%s api_base=%s", args.llm_model, args.api_base)
        print(f"生成回答失败: {describe_llm_error(exc)}", file=sys.stderr)
        # 检索阶段已经完成；把片段打印出来，便于判断失败前证据是否已经取到、
        # 内容是否足以回答，而不是让用户只能重跑一次检索。
        print("\n检索已完成，以下证据已取得但未生成回答，可供排查:")
        _print_evidence(results, set())
        return 1

    print(f"\n回答:\n{answer.answer}")
    _print_citation_check(answer)
    _print_evidence(answer.evidence, set(answer.citations) - set(answer.invalid_citations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
