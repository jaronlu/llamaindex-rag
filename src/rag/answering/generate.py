"""用检索片段生成带引用的回答，并检查引用编号是否可以核对。

本模块不负责检索：调用方先通过 ``retrieval.search`` 拿到已经过来源校验的片段，
再交给这里编号、拼装提示词并调用生成模型。回答中的 ``[n]`` 必须对应交付列表里的
片段编号；编号越界或完全没有引用都会作为结果字段返回，由命令行明确提示，避免把
“生成了回答”当成“回答可核对”。
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from ..config.answering import DEFAULT_MIN_DENSE_SCORE, DEFAULT_MIN_RERANK_SCORE
from ..retrieval.search import RetrievalResult

if TYPE_CHECKING:
    from llama_index.llms.openai_like import OpenAILike  # type: ignore[import-untyped]


# 只认 ASCII 方括号加数字。PDF 原文里也可能出现 `[1]` 这类编号（例如法条编号），
# 被模型照抄时会误判为引用并触发“不存在的证据编号”提示；目前只提示、不阻塞，
# 这一误报边界记录在 docs/knowledge/07-answer-issues.md。
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

INSUFFICIENT_EVIDENCE_ANSWER = "未在文档中检索到与问题足够相关的内容，无法基于资料回答。"

SYSTEM_PROMPT = (
    "你是文档问答助手。只能依据用户给出的证据片段回答，不得使用证据之外的知识、"
    "常识或猜测。回答中的每个关键结论后面用 [n] 标注证据编号，n 必须是提供的编号。"
    "如果证据不足以回答，直接说明资料中没有明确答案，并指出缺少什么信息，不要拼凑。"
    "不要编造页码、文件名或原文。用简洁的中文回答。"
)


class ChatModel(Protocol):
    """生成模型的最小接口，便于测试替换；生产环境由 ``OpenAILike`` 实现。"""

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse: ...


class StreamingChatModel(ChatModel, Protocol):
    """在 ``ChatModel`` 基础上增加流式输出；仅流式接口需要这个能力。

    分开定义是为了让只做整段生成的调用方和测试替身不必实现 ``stream_chat``。
    """

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[ChatResponse]: ...


@dataclass(frozen=True, slots=True)
class Answer:
    """一次基于来源的回答及其引用检查结果。

    ``evidence`` 的顺序就是提示词中的编号顺序，第 n 项对应 ``[n]``。
    ``invalid_citations`` 保存回答中出现但不对应任何证据的编号；它们不能作为
    依据，CLI 必须展示警告。``llm_called`` 为 ``False`` 表示没有可交付片段，
    程序直接给出资料不足提示，没有调用外部模型。
    """

    question: str
    answer: str
    evidence: tuple[RetrievalResult, ...]
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    llm_called: bool


def resolve_api_key(env_var: str) -> str:
    """从环境变量读取密钥；缺失时只说明变量名，不回显任何取值。

    命令行和局域网接口都要从环境变量取密钥，集中在这里可以保证两处的报错措辞
    和“不打印密钥”的约定一致，也便于测试替换环境变量。
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ValueError(f"未设置环境变量 {env_var}；密钥只从环境变量读取，不写入命令行、配置或日志")
    return key


def create_llm(
    *,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> OpenAILike:
    """创建指向 OpenAI 兼容服务的聊天模型。

    通过 ``OpenAILike`` 复用 LlamaIndex 的 LLM 接口；本项目只封装客户端构造和
    参数默认值，不实现协议本身。API key 由调用方从环境变量读入，函数不记录它。
    """
    from llama_index.llms.openai_like import OpenAILike  # type: ignore[import-untyped]

    return OpenAILike(
        model=model,
        api_base=api_base,
        api_key=api_key,
        is_chat_model=True,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def select_relevant_evidence(
    results: Sequence[RetrievalResult],
    *,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
) -> list[RetrievalResult]:
    """按相关性分数过滤证据，并把保留项重新编号。

    重排分和 dense 相似度的量纲不同，分别使用不同阈值。结果是否经过重排，以第一条
    是否带 rerank_score 判断；缺失分数的片段无法证明相关，直接丢弃。返回的 rank
    从 1 连续编号，避免过滤后“证据 2”对应回答里的 [1] 这种错位。
    """
    if not results:
        return []
    reranked = results[0].rerank_score is not None
    threshold = min_rerank_score if reranked else min_dense_score
    selected: list[RetrievalResult] = []
    for result in results:
        score = result.rerank_score if reranked else result.score
        if score is None or score < threshold:
            continue
        selected.append(replace(result, rank=len(selected) + 1))
    return selected


def format_evidence(results: Sequence[RetrievalResult]) -> str:
    """把候选片段编号成提示词中的证据块。

    每个块都带文件名和物理页码，正文完整保留，不截断；模型看到的证据与用户
    随后能核对的内容一致，避免提示词里丢失部分原文却照常生成回答。
    """
    blocks: list[str] = []
    for number, result in enumerate(results, start=1):
        blocks.append(
            f"[{number}] 文件: {result.file_name}；物理页码: {result.page_label}\n"
            f"正文:\n{result.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: Sequence[RetrievalResult]) -> list[ChatMessage]:
    """构造系统约束和带证据的用户消息。"""
    if not results:
        raise ValueError("没有可用的证据片段，不能构造回答请求")
    user_content = f"问题：{question.strip()}\n\n证据：\n\n{format_evidence(results)}"
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]


def _response_truncated(response: ChatResponse) -> bool:
    """判断服务端是否因输出上限截断回答。

    OpenAI 兼容服务把结束原因放在原始响应里；``ChatResponse`` 没有统一字段，
    这里兼容字典和 SDK 对象两种形态。识别失败按未截断处理，但调用方仍会看到
    空回答等其它错误，不会静默把异常当成功。
    """
    raw = response.raw
    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return False
    first = choices[0]
    reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
    return reason == "length"


def parse_citations(text: str) -> tuple[int, ...]:
    """按出现顺序提取 ``[n]`` 编号，重复编号只保留一次。"""
    seen: set[int] = set()
    citations: list[int] = []
    for match in CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            citations.append(number)
    return tuple(citations)


def _answer_from_content(
    question: str,
    evidence: tuple[RetrievalResult, ...],
    content: object,
) -> Answer:
    """根据完整回答文本构造结果，并完成引用编号检查。

    流式和非流式回答都走这一处：引用解析、越界编号和空回答判定不能各写一份，
    否则同一个模型输出在两种接口下会得到不同结论。
    """
    if content is None or (isinstance(content, str) and not content.strip()):
        raise ValueError("生成模型返回了空回答")
    # 部分 OpenAI 兼容服务会返回 content blocks 等非字符串结构；直接报“空回答”
    # 会掩盖真实原因，这里单独给出可定位的错误。
    if not isinstance(content, str):
        raise ValueError(f"生成模型返回了非文本内容（{type(content).__name__}），当前仅支持纯文本回答")
    citations = parse_citations(content)
    invalid = tuple(number for number in citations if number < 1 or number > len(evidence))
    return Answer(
        question=question.strip(),
        answer=content.strip(),
        evidence=evidence,
        citations=citations,
        invalid_citations=invalid,
        llm_called=True,
    )


def answer_from_text(
    question: str,
    results: Sequence[RetrievalResult],
    content: str,
) -> Answer:
    """用已经收集到的流式文本构造与 ``generate_answer`` 一致的 ``Answer``。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    evidence = tuple(results)
    if not evidence:
        raise ValueError("没有证据片段，不能构造回答结果")
    return _answer_from_content(question, evidence, content)


def iter_answer_deltas(
    question: str,
    results: Sequence[RetrievalResult],
    llm: StreamingChatModel,
) -> Iterator[str]:
    """逐段产出生成文本，供流式接口立即转发给客户端。

    只转发最终回答的增量，不包含模型的推理内容；调用方负责把文本拼接起来，
    最后交给 ``answer_from_text`` 做引用检查。输出被截断或完全没有文本时在
    生成器结束处抛 ``ValueError``，调用方必须把它当作失败，而不是接受半截回答。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    evidence = tuple(results)
    if not evidence:
        raise ValueError("没有证据片段，不能请求生成模型")

    truncated = False
    emitted = False
    for chunk in llm.stream_chat(build_messages(question, evidence)):
        # 结束原因可能只在最后一个 chunk 上出现，因此每个 chunk 都要检查。
        truncated = _response_truncated(chunk) or truncated
        delta = chunk.delta
        if not delta:
            continue
        emitted = True
        yield delta

    if truncated:
        # 已经发出的半截文本不能被调用方当作完整回答；顺序上先检查截断再检查空文本，
        # 让“输出达到上限”这个更具体的原因不被“空回答”掩盖。
        raise ValueError("生成模型输出达到 max_tokens 上限，回答不完整；请调大上限或减少证据数量")
    if not emitted:
        raise ValueError("生成模型返回了空回答")


def generate_answer(
    question: str,
    results: Sequence[RetrievalResult],
    llm: ChatModel | None = None,
) -> Answer:
    """基于给定片段调用生成模型，并记录引用编号。

    ``results`` 视为已经按交付顺序排好，编号从 1 开始。没有片段时不需要 ``llm``，
    直接返回资料不足提示；这样既不会让模型对着空证据硬答，也不会产生外部调用。
    有片段却没有提供模型时属于调用错误。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    evidence = tuple(results)
    if not evidence:
        return Answer(
            question=question.strip(),
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            evidence=(),
            citations=(),
            invalid_citations=(),
            llm_called=False,
        )
    if llm is None:
        raise ValueError("有证据片段时必须提供生成模型")

    response = llm.chat(build_messages(question, evidence))
    if _response_truncated(response):
        # 截断的回答会停在句子中间，继续引用检查会让人误以为它只是“没引用全”，
        # 所以直接报失败，不交付不完整答案。
        raise ValueError("生成模型输出达到 max_tokens 上限，回答不完整；请调大上限或减少证据数量")
    return _answer_from_content(question, evidence, response.message.content)
