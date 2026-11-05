"""连续对话命令行：以索引为入口，启动后只输入中文问题即可反复问答。

本命令负责把已有模块串成一个交互循环：检索和重排只加载一次，每轮调用
RetrievalSession.search；有历史时先把追问改写成独立问题，再按同一套相关度
门槛筛选证据并生成带引用的回答。命令不重新实现检索、生成或引用检查规则，输出
与单次回答命令保持一致；默认只显示来源行，加 --show-evidence 才打印完整原文。

外部模型调用只发生在改写和生成阶段，问题与证据会发送到配置的服务并产生费用；
密钥仍只从环境变量读取。对话可以一直进行，但上下文只保留最近若干轮，避免
提示词和成本无限增长。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .arguments import add_retrieval_arguments
from .logging import logged_command
from .output import print_citation_check, print_evidence

from ..answering.conversation import ChatTurn, condense_question
from ..answering.generate import (
    ChatModel,
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
from ..config.chat import (
    DEFAULT_CHAT_CONDENSE_MAX_TOKENS,
    DEFAULT_CHAT_CONDENSE_TEMPERATURE,
    DEFAULT_CHAT_HISTORY_TURNS,
)
from ..config.retrieval import DEFAULT_RETRIEVAL_TOP_K
from ..retrieval.search import RetrievalSession


logger = logging.getLogger(__name__)

_EXIT_COMMANDS = {":quit", ":exit", "退出", "exit", "q"}
_CLEAR_COMMANDS = {":clear", "清空"}
_HELP_COMMANDS = {":help", "帮助"}
_PROMPT = "问题> "


def _print_help() -> None:
    """打印交互命令；中文问题本身不会和这些命令冲突。"""
    print("输入中文问题开始问答。")
    print("  :quit / 退出   结束对话")
    print("  :clear / 清空  清空最近对话上下文")
    print("  :help / 帮助   显示本帮助")


def _run_turn(
    question: str,
    history: list[ChatTurn],
    session: RetrievalSession,
    answer_llm: ChatModel,
    rewrite_llm: ChatModel,
    args: argparse.Namespace,
) -> ChatTurn | None:
    """完成一轮改写、检索、过滤和回答，返回可加入历史的问答。

    返回 None 表示检索或生成失败，本轮不计入历史，避免失败结果影响后续
    追问改写。失败只结束当前轮，不退出整个对话。
    """
    search_question = question
    if args.history_turns > 0 and history:
        try:
            search_question = condense_question(question, history[-args.history_turns:], rewrite_llm)
        except Exception as exc:
            logger.warning("chat.condense_failed question_length=%s", len(question))
            print(f"追问改写失败，按原问题检索: {describe_llm_error(exc)}")
            search_question = question

    try:
        results = session.search(search_question)
    except Exception as exc:
        logger.exception("chat.retrieval_failed question_length=%s", len(search_question))
        print(f"检索失败: {exc}", file=sys.stderr)
        return None

    candidate_count = len(results)
    results = select_relevant_evidence(
        results,
        min_rerank_score=args.min_rerank_score,
        min_dense_score=args.min_dense_score,
    )

    print(f"\n问题: {question}")
    if search_question != question:
        print(f"检索用问题: {search_question}")
    if candidate_count != len(results):
        print(f"相关度过滤: {candidate_count} 条候选中 {candidate_count - len(results)} 条低于门槛，未作为证据。")

    if not results:
        answer = generate_answer(search_question, results)
        print("资料不足: 没有候选片段通过来源、正文和相关度检查，未调用生成模型。")
        print(f"回答: {answer.answer}")
        return ChatTurn(question=search_question, answer=answer.answer)

    try:
        answer = generate_answer(search_question, results, answer_llm)
    except Exception as exc:
        logger.exception(
            "chat.generation_failed model=%s question_length=%s evidence=%s",
            args.llm_model,
            len(search_question),
            len(results),
        )
        print(f"生成回答失败: {describe_llm_error(exc)}", file=sys.stderr)
        print("\n检索已完成，以下证据已取得但未生成回答，可供排查:")
        print_evidence(results, set())
        return None

    print(f"\n回答:\n{answer.answer}")
    print_citation_check(answer)
    print_evidence(
        answer.evidence,
        set(answer.citations) - set(answer.invalid_citations),
        show_text=args.show_evidence,
    )
    return ChatTurn(question=search_question, answer=answer.answer)


@logged_command
def main(argv: list[str] | None = None) -> int:
    """启动连续对话；索引是唯一必填入口，其余参数都有项目默认值。"""
    parser = argparse.ArgumentParser(description="以索引为入口启动连续中文问答")
    parser.add_argument("index", type=Path, help="索引目录")
    add_retrieval_arguments(
        parser,
        top_k_help=f"每轮作为证据交付的片段数量（默认 {DEFAULT_RETRIEVAL_TOP_K}）",
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
    parser.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_CHAT_HISTORY_TURNS,
        help=f"保留最近多少轮问答用于改写追问；0 表示关闭上下文（默认 {DEFAULT_CHAT_HISTORY_TURNS}）",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="每轮显示完整证据原文；默认只显示来源、页码和来源状态",
    )
    args = parser.parse_args(argv)

    if args.history_turns < 0:
        print("启动失败: --history-turns 不能为负数", file=sys.stderr)
        return 1

    # 先检查密钥再加载模型：密钥缺失时没有必要花时间加载数 GB 的重排模型。
    try:
        api_key = resolve_api_key(args.api_key_env)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    print("正在加载索引和本地模型；首次启用重排时可能需要数十秒...")
    try:
        session = RetrievalSession(
            args.index,
            args.model_path,
            args.rerank_model_path,
            dense_only=args.dense_only,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
    except Exception as exc:
        logger.exception("chat.startup_failed index=%s", args.index)
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    try:
        answer_llm = create_llm(
            model=args.llm_model,
            api_base=args.api_base,
            api_key=api_key,
            temperature=DEFAULT_ANSWER_TEMPERATURE,
            max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
            timeout=DEFAULT_ANSWER_TIMEOUT_SECONDS,
            trust_env=args.llm_trust_env,
        )
        # 改写只输出一个短问题，单独使用更小的输出上限和更低的温度；两个客户端
        # 共享同一份密钥和地址，不额外增加配置。
        rewrite_llm = create_llm(
            model=args.llm_model,
            api_base=args.api_base,
            api_key=api_key,
            temperature=DEFAULT_CHAT_CONDENSE_TEMPERATURE,
            max_tokens=DEFAULT_CHAT_CONDENSE_MAX_TOKENS,
            timeout=DEFAULT_ANSWER_TIMEOUT_SECONDS,
            trust_env=args.llm_trust_env,
        )
    except Exception as exc:
        print(f"启动失败: 创建生成模型失败: {describe_llm_error(exc)}", file=sys.stderr)
        return 1

    print(f"索引: {args.index.expanduser().resolve()}")
    print(
        "检索模式: dense"
        if args.dense_only
        else f"检索模式: dense top {args.candidate_k} → rerank top {args.top_k}"
    )
    print(f"生成模型: {args.llm_model}（{args.api_base}）")
    print(f"相关度门槛: dense ≥ {args.min_dense_score} / rerank ≥ {args.min_rerank_score}")
    if args.history_turns:
        print(f"多轮上下文: 最近 {args.history_turns} 轮；追问会先改写成独立问题再检索")
    else:
        print("多轮上下文: 已关闭；每个问题独立检索")
    print(f"生成请求代理: {'跟随系统和环境变量' if args.llm_trust_env else '直连（--no-llm-proxy）'}")
    print("外部调用: 每轮会把问题和通过门槛的片段发送到上述地址并产生费用；不会发送 PDF、索引或其他文档。")
    if not args.show_evidence:
        print("提示: 默认只显示来源行；加 --show-evidence 可查看完整证据原文。")
    print("输入中文问题开始对话；:quit 退出，:clear 清空上下文，:help 查看帮助。")

    history: list[ChatTurn] = []
    while True:
        try:
            raw = input(_PROMPT)
        except EOFError:
            print("\n已退出。")
            return 0
        except KeyboardInterrupt:
            print("\n已退出。")
            return 0

        question = raw.strip()
        if not question:
            continue
        if question in _EXIT_COMMANDS:
            print("已退出。")
            return 0
        if question in _CLEAR_COMMANDS:
            history.clear()
            print("已清空对话上下文。")
            continue
        if question in _HELP_COMMANDS:
            _print_help()
            continue

        turn = _run_turn(question, history, session, answer_llm, rewrite_llm, args)
        if turn is None:
            continue
        if args.history_turns:
            history.append(turn)
            if len(history) > args.history_turns:
                del history[:-args.history_turns]


if __name__ == "__main__":
    sys.exit(main())
