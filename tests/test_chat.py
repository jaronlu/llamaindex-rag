import contextlib
import io
import os
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from src.rag.cli.chat import main
from src.rag.retrieval.search import RetrievalResult


def setUpModule():
    """把对话测试的日志写入临时文件，避免污染用户的运行日志。"""
    temp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(temp.cleanup)
    env = patch.dict(os.environ, {"RAG_LOG_FILE": str(Path(temp.name) / "test.log"), "RAG_LOG_LEVEL": "INFO"})
    env.start()
    unittest.addModuleCleanup(env.stop)


def make_result(text: str = "加班费按三档支付。") -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        score=0.9,
        node_id="node-1",
        text=text,
        file_name="HowToLiveBetter.pdf",
        source_path="/docs/pdf/HowToLiveBetter.pdf",
        page_numbers=(291, 292),
        source_exists=True,
        provenance_complete=True,
        rerank_score=0.8,
    )


class FakeModel:
    """返回固定文本并记录收到的消息。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        self.messages.append(list(messages))
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.text),
            raw={},
        )


class FakeSession:
    """替身检索会话，记录每轮实际检索的问题。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.questions: list[str] = []

    def search(self, question: str) -> list[RetrievalResult]:
        self.questions.append(question)
        return self.results


class ChatCommandTests(unittest.TestCase):
    def run_chat(
        self,
        inputs: list[str],
        *,
        results: list[RetrievalResult] | None = None,
        answer_text: str = "按三档支付 [1]。",
        rewrite_text: str = "改写后的独立问题？",
        extra_args: list[str] | None = None,
    ):
        """在替身模型和替身检索会话下运行一轮或多轮对话。"""
        session = FakeSession([make_result()] if results is None else results)
        answer_model = FakeModel(answer_text)
        rewrite_model = FakeModel(rewrite_text)
        with (
            patch("src.rag.cli.chat.RetrievalSession", return_value=session),
            patch("src.rag.cli.chat.create_llm", side_effect=[answer_model, rewrite_model]),
            patch("src.rag.cli.chat.resolve_api_key", return_value="test-key"),
            patch("builtins.input", side_effect=inputs),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = main(["index", *(extra_args or [])])
        return code, output.getvalue(), session, answer_model, rewrite_model

    def test_answers_one_question_then_quits(self):
        code, output, session, answer_model, rewrite_model = self.run_chat(["加班费怎么算？", ":quit"])

        self.assertEqual(code, 0)
        self.assertEqual(session.questions, ["加班费怎么算？"])
        self.assertEqual(len(answer_model.messages), 1)
        self.assertEqual(rewrite_model.messages, [])
        self.assertIn("回答:\n按三档支付 [1]。", output)
        self.assertIn("引用检查: 回答引用了 [1]", output)
        self.assertIn("物理页码: 291–292", output)
        self.assertIn("输入中文问题开始对话", output)
        # 默认只显示来源行，不把完整证据正文刷满终端。
        self.assertNotIn("完整原文:", output)

    def test_show_evidence_prints_full_text(self):
        code, output, _, _, _ = self.run_chat(
            ["加班费怎么算？", ":quit"],
            extra_args=["--show-evidence"],
        )

        self.assertEqual(code, 0)
        self.assertIn("完整原文:", output)
        self.assertIn("加班费按三档支付。", output)

    def test_follow_up_is_rewritten_before_retrieval(self):
        code, output, session, _, rewrite_model = self.run_chat(
            ["年休假有多少天？", "那加班费呢？", ":quit"],
            rewrite_text="加班费怎么计算？",
        )

        self.assertEqual(code, 0)
        self.assertEqual(session.questions, ["年休假有多少天？", "加班费怎么计算？"])
        self.assertEqual(len(rewrite_model.messages), 1)
        self.assertIn("检索用问题: 加班费怎么计算？", output)

    def test_insufficient_evidence_skips_generation(self):
        code, output, _, answer_model, _ = self.run_chat(
            ["这本书讲了什么？", ":quit"],
            results=[],
        )

        self.assertEqual(code, 0)
        self.assertEqual(answer_model.messages, [])
        self.assertIn("资料不足", output)

    def test_clear_resets_history(self):
        code, output, session, _, rewrite_model = self.run_chat(
            ["年休假？", ":clear", "加班费？", ":quit"],
        )

        self.assertEqual(code, 0)
        self.assertEqual(session.questions, ["年休假？", "加班费？"])
        self.assertEqual(rewrite_model.messages, [])
        self.assertIn("已清空对话上下文", output)

    def test_history_turns_zero_disables_rewrite(self):
        code, output, session, _, rewrite_model = self.run_chat(
            ["年休假？", "加班费？", ":quit"],
            extra_args=["--history-turns", "0"],
        )

        self.assertEqual(code, 0)
        self.assertEqual(session.questions, ["年休假？", "加班费？"])
        self.assertEqual(rewrite_model.messages, [])
        self.assertIn("多轮上下文: 已关闭", output)

    def test_missing_api_key_fails_before_loading_models(self):
        with (
            patch("src.rag.cli.chat.RetrievalSession") as session,
            patch("src.rag.cli.chat.resolve_api_key", side_effect=ValueError("未设置环境变量 KEY")),
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            code = main(["index"])

        self.assertEqual(code, 1)
        session.assert_not_called()
        self.assertIn("启动失败", error.getvalue())
        self.assertIn("未设置环境变量 KEY", error.getvalue())

    def test_index_or_model_failure_reports_startup_error(self):
        with patch("src.rag.cli.chat.RetrievalSession", side_effect=ValueError("索引目录不存在")), \
             patch("src.rag.cli.chat.resolve_api_key", return_value="test-key"), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(["missing-index"])

        self.assertEqual(code, 1)
        self.assertIn("启动失败", error.getvalue())
        self.assertIn("索引目录不存在", error.getvalue())

    def test_negative_history_turns_is_rejected(self):
        with (
            patch("src.rag.cli.chat.RetrievalSession") as session,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            code = main(["index", "--history-turns", "-1"])

        self.assertEqual(code, 1)
        session.assert_not_called()
        self.assertIn("不能为负数", error.getvalue())


if __name__ == "__main__":
    unittest.main()
