"""用检索片段生成带引用的回答，并检查引用编号是否可以核对。

本模块不负责检索：调用方先通过 ``retrieval.search`` 拿到已经过来源校验的片段，
再交给这里编号、拼装提示词并调用生成模型。回答中的 ``[n]`` 必须对应交付列表里的
片段编号；编号越界或完全没有引用都会作为结果字段返回，由命令行明确提示，避免把
“生成了回答”当成“回答可核对”。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from ..retrieval.search import RetrievalResult

if TYPE_CHECKING:
    from llama_index.llms.openai_like import OpenAILike  # type: ignore[import-untyped]


CITATION_PATTERN = re.compile(r"\[(\d+)\]")

INSUFFICIENT_EVIDENCE_ANSWER = "未在文档中检索到可交付的片段，无法基于资料回答。"

SYSTEM_PROMPT = (
    "你是文档问答助手。只能依据用户给出的证据片段回答，不得使用证据之外的知识、"
    "常识或猜测。回答中的每个关键结论后面用 [n] 标注证据编号，n 必须是提供的编号。"
    "如果证据不足以回答，直接说明资料中没有明确答案，并指出缺少什么信息，不要拼凑。"
    "不要编造页码、文件名或原文。用简洁的中文回答。"
)


class ChatModel(Protocol):
    """生成模型的最小接口，便于测试替换；生产环境由 ``OpenAILike`` 实现。"""

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse: ...


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
    content = response.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("生成模型返回了空回答")
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
