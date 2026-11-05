"""连续对话中的追问改写，把依赖上文的问题变成可独立检索的问题。

问题改写是检索前的一步：用户说“那它要多少天？”时，直接拿这句话去检索通常
得不到正确片段。这里用生成模型结合最近几轮问答改写问题，然后仍交给原有检索、
重排、相关度过滤和引用检查；不改写答案本身，也不把历史直接当证据。

改写失败不终止对话：调用方应回退到原问题继续检索。模型可能改写过度或引入
原文没有的限定，因此命令行会把改写后的问题显示出来，方便对照。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from ..config.chat import DEFAULT_CHAT_CONTEXT_ANSWER_CHARS
from .generate import ChatModel, response_truncated


CONDENSE_SYSTEM_PROMPT = (
    "你是对话中的问题改写助手。根据最近对话，把用户的最新问题改写成不依赖上文、"
    "可以单独用于文档检索的完整中文问题。保留原问题中的专有名词、限定条件和时间；"
    "不要回答问题，不要添加对话中没有的信息。如果最新问题已经可以独立理解，原样输出。"
    "只输出改写后的问题，不要输出解释、前缀或引号。"
)

_PREFIXES = ("改写后的问题：", "改写后问题：", "改写的问题：", "问题：")
_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ('"', '"'), ("'", "'"))


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """一轮已经完成的问答，供下一轮改写追问时参考。

    只保存原始问题和回答正文；证据、分数和来源不进入改写提示词，避免把整段
    文档内容重复发送给模型。
    """

    question: str
    answer: str


def _history_text(history: Sequence[ChatTurn], answer_char_limit: int) -> str:
    """把历史问答拼成改写提示词中的上下文块，并限制单条回答长度。"""
    blocks: list[str] = []
    for turn in history:
        answer = turn.answer.strip()
        if len(answer) > answer_char_limit:
            answer = answer[:answer_char_limit] + "……（后文略）"
        blocks.append(f"用户：{turn.question.strip()}\n助手：{answer}")
    return "\n\n".join(blocks)


def build_condense_messages(
    question: str,
    history: Sequence[ChatTurn],
    *,
    answer_char_limit: int = DEFAULT_CHAT_CONTEXT_ANSWER_CHARS,
) -> list[ChatMessage]:
    """构造把追问改写成独立问题的提示词。

    历史为空时没有可改写的上下文，调用方应先原样使用当前问题；这里仍返回
    一条明确要求原样输出的消息，方便单独检查提示词结构。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if answer_char_limit <= 0:
        raise ValueError("历史回答长度上限必须是正整数")
    context = _history_text(history, answer_char_limit) if history else "（没有更早的对话）"
    user_content = (
        f"最近对话：\n\n{context}\n\n"
        f"最新问题：{question.strip()}\n\n"
        "请输出适合单独检索的中文问题；已经独立时原样输出。只输出问题本身。"
    )
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=CONDENSE_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]


def _normalize_rewritten(text: str) -> str:
    """去掉模型偶尔附带的问题前缀和成对引号。"""
    result = text.strip()
    for prefix in _PREFIXES:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
            break
    for left, right in _QUOTE_PAIRS:
        if len(result) >= 2 and result.startswith(left) and result.endswith(right):
            result = result[1:-1].strip()
            break
    return result


def condense_question(
    question: str,
    history: Sequence[ChatTurn],
    llm: ChatModel,
    *,
    answer_char_limit: int = DEFAULT_CHAT_CONTEXT_ANSWER_CHARS,
) -> str:
    """用最近对话把追问改写成独立问题；没有历史时直接返回原问题。

    返回空文本、模型输出被截断或不是字符串时抛 ValueError。调用方可以捕获
    异常并回退到原问题，不应把半截改写当成完整问题继续检索。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if not history:
        return question.strip()

    response = llm.chat(build_condense_messages(question, history, answer_char_limit=answer_char_limit))
    if response_truncated(response):
        raise ValueError("追问改写输出达到 max_tokens 上限，问题不完整")
    content = response.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("追问改写模型返回了空内容")
    rewritten = _normalize_rewritten(content)
    if not rewritten:
        raise ValueError("追问改写模型返回了空问题")
    return rewritten
