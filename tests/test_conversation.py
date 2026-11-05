import unittest
from collections.abc import Sequence

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from src.rag.answering.conversation import (
    CONDENSE_SYSTEM_PROMPT,
    ChatTurn,
    build_condense_messages,
    condense_question,
)


class FakeModel:
    """记录收到的消息并返回固定文本，供改写测试使用。"""

    def __init__(self, text: str, raw: object = None) -> None:
        self.text = text
        self.raw = raw
        self.messages: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        self.messages.append(list(messages))
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.text),
            raw=self.raw,
        )


class CondensePromptTests(unittest.TestCase):
    def test_builds_system_and_user_messages_with_history(self):
        history = [
            ChatTurn(question="年休假有多少天？", answer="按累计工龄分为 5、10、15 天。"),
            ChatTurn(question="那加班费呢？", answer="加班费按三档支付。"),
        ]
        messages = build_condense_messages("两者能同时享受吗？", history)

        self.assertEqual(messages[0].role, MessageRole.SYSTEM)
        self.assertEqual(messages[0].content, CONDENSE_SYSTEM_PROMPT)
        self.assertEqual(messages[1].role, MessageRole.USER)
        user = messages[1].content or ""
        self.assertIn("用户：年休假有多少天？", user)
        self.assertIn("助手：按累计工龄分为 5、10、15 天。", user)
        self.assertIn("用户：那加班费呢？", user)
        self.assertIn("最新问题：两者能同时享受吗？", user)

    def test_limits_long_answers_in_history(self):
        long_answer = "很长的回答" * 100
        messages = build_condense_messages(
            "继续",
            [ChatTurn(question="问题", answer=long_answer)],
            answer_char_limit=10,
        )
        user = messages[1].content or ""
        self.assertIn("助手：很长的回答很长的回答……（后文略）", user)
        self.assertIn("后文略", user)
        self.assertNotIn(long_answer, user)

    def test_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            build_condense_messages("  ", [])

    def test_rejects_non_positive_answer_limit(self):
        with self.assertRaisesRegex(ValueError, "正整数"):
            build_condense_messages("问题", [], answer_char_limit=0)


class CondenseQuestionTests(unittest.TestCase):
    def test_returns_original_question_without_history_and_skips_model(self):
        model = FakeModel("不应调用")
        self.assertEqual(condense_question("  年休假有多少天？  ", [], model), "年休假有多少天？")
        self.assertEqual(model.messages, [])

    def test_rewrites_follow_up_through_model(self):
        model = FakeModel("年休假按累计工龄有多少天？")
        result = condense_question("那它要多少天？", [ChatTurn("年休假是什么？", "是一种带薪假期。")], model)

        self.assertEqual(result, "年休假按累计工龄有多少天？")
        self.assertEqual(len(model.messages), 1)

    def test_strips_common_prefix_and_quotes(self):
        model = FakeModel("改写后的问题：“年休假有多少天？”")
        result = condense_question("那它呢？", [ChatTurn("年休假", "带薪假期")], model)
        self.assertEqual(result, "年休假有多少天？")

    def test_rejects_empty_rewrite(self):
        with self.assertRaisesRegex(ValueError, "空内容"):
            condense_question("追问", [ChatTurn("问题", "回答")], FakeModel("   "))

    def test_rejects_truncated_rewrite(self):
        model = FakeModel("半截问题", raw={"choices": [{"finish_reason": "length"}]})
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            condense_question("追问", [ChatTurn("问题", "回答")], model)

    def test_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            condense_question(" ", [ChatTurn("问题", "回答")], FakeModel("改写"))


if __name__ == "__main__":
    unittest.main()
