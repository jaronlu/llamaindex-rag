import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from collections.abc import Sequence

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from src.rag.answering.generate import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    build_messages,
    create_llm,
    generate_answer,
    parse_citations,
)
from src.rag.cli.answering import main
from src.rag.config.answering import DEFAULT_ANSWER_API_KEY_ENV
from src.rag.retrieval.search import RetrievalResult


def make_result(
    rank: int,
    text: str,
    pages: tuple[int, ...] = (291, 292),
    provenance_complete: bool = True,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=0.9,
        node_id=f"node-{rank}",
        text=text,
        file_name="HowToLiveBetter.pdf",
        source_path="/docs/pdf/HowToLiveBetter.pdf",
        page_numbers=pages,
        source_exists=True,
        provenance_complete=provenance_complete,
        provenance_warning=None if provenance_complete else "来源文件内容已变化",
    )


class FakeModel:
    """返回固定文本的聊天模型，并记录收到的消息。"""

    def __init__(self, text: str, raw: object = None) -> None:
        self.text = text
        self.raw = raw
        self.messages: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        self.messages.append(list(messages))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.text), raw=self.raw)


class CitationParsingTests(unittest.TestCase):
    def test_extracts_numbers_in_order_and_deduplicates(self):
        self.assertEqual(parse_citations("见 [2]，另见 [1] 和 [2]。"), (2, 1))

    def test_returns_empty_without_citations(self):
        self.assertEqual(parse_citations("资料中没有明确答案。"), ())

    def test_keeps_out_of_range_numbers_for_later_validation(self):
        self.assertEqual(parse_citations("[0] [9]"), (0, 9))


class PromptTests(unittest.TestCase):
    def test_messages_include_constraints_and_full_evidence(self):
        results = [
            make_result(1, "加班费按工作日、休息日、法定节假日三档支付。"),
            make_result(2, "年休假按累计工龄分别为 5、10、15 天。", pages=(360, 361)),
        ]
        messages = build_messages("加班费和年休假怎么算？", results)

        self.assertEqual(messages[0].role, MessageRole.SYSTEM)
        self.assertEqual(messages[0].content, SYSTEM_PROMPT)
        self.assertIn("只能依据", messages[0].content or "")
        self.assertEqual(messages[1].role, MessageRole.USER)
        user = messages[1].content or ""
        self.assertIn("加班费和年休假怎么算？", user)
        self.assertIn("[1] 文件: HowToLiveBetter.pdf；物理页码: 291–292", user)
        self.assertIn("[2] 文件: HowToLiveBetter.pdf；物理页码: 360–361", user)
        self.assertIn("加班费按工作日、休息日、法定节假日三档支付。", user)
        self.assertIn("年休假按累计工龄分别为 5、10、15 天。", user)

    def test_rejects_empty_evidence(self):
        with self.assertRaises(ValueError):
            build_messages("问题", [])


class GenerateAnswerTests(unittest.TestCase):
    def test_records_valid_citations(self):
        results = [make_result(1, "正文一"), make_result(2, "正文二")]
        model = FakeModel("按三档支付 [1]，年休假按工龄 [2][1]。")

        answer = generate_answer("问题", results, model)

        self.assertTrue(answer.llm_called)
        self.assertEqual(answer.citations, (1, 2))
        self.assertEqual(answer.invalid_citations, ())
        self.assertEqual(answer.evidence, tuple(results))
        self.assertEqual(answer.answer, "按三档支付 [1]，年休假按工龄 [2][1]。")
        self.assertEqual(len(model.messages), 1)

    def test_flags_citations_outside_evidence(self):
        results = [make_result(1, "正文一")]
        answer = generate_answer("问题", results, FakeModel("结论 [1]，另见 [3]。"))

        self.assertEqual(answer.citations, (1, 3))
        self.assertEqual(answer.invalid_citations, (3,))

    def test_empty_evidence_skips_model(self):
        model = FakeModel("不应被调用")

        answer = generate_answer("问题", [], model)

        self.assertFalse(answer.llm_called)
        self.assertEqual(answer.answer, INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertEqual(answer.citations, ())
        self.assertEqual(model.messages, [])

    def test_requires_model_when_evidence_exists(self):
        with self.assertRaises(ValueError):
            generate_answer("问题", [make_result(1, "正文")])

    def test_rejects_truncated_answer(self):
        model = FakeModel("回答写到一半", raw={"choices": [{"finish_reason": "length"}]})

        with self.assertRaisesRegex(ValueError, "max_tokens"):
            generate_answer("问题", [make_result(1, "正文")], model)

    def test_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            generate_answer("问题", [make_result(1, "正文")], FakeModel("   "))

    def test_rejects_non_text_content_with_specific_message(self):
        class NonTextModel:
            def chat(self, messages):
                return SimpleNamespace(raw={}, message=SimpleNamespace(content=[{"type": "text", "text": "回答"}]))

        with self.assertRaisesRegex(ValueError, "非文本内容"):
            generate_answer("问题", [make_result(1, "正文")], NonTextModel())

    def test_detects_truncation_from_sdk_style_raw_response(self):
        class Choice:
            finish_reason = "length"

        class Raw:
            choices = [Choice()]

        with self.assertRaisesRegex(ValueError, "max_tokens"):
            generate_answer("问题", [make_result(1, "正文")], FakeModel("回答写到一半", raw=Raw()))

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            generate_answer(" ", [make_result(1, "正文")], FakeModel("回答"))


class CreateLlmTests(unittest.TestCase):
    def test_builds_openai_compatible_chat_model(self):
        llm = create_llm(
            model="glm-5.3-flash",
            api_base="https://example.com/v3",
            # 只用于构造客户端对象，不会发起请求；用非字面量避免被误判为真实密钥。
            api_key="x" * 32,
            temperature=0.1,
            max_tokens=512,
            timeout=30.0,
        )

        self.assertEqual(llm.model, "glm-5.3-flash")
        self.assertEqual(llm.api_base, "https://example.com/v3")
        self.assertTrue(llm.is_chat_model)
        self.assertEqual(llm.max_tokens, 512)


class AnswerCommandTests(unittest.TestCase):
    def test_missing_api_key_fails_without_echoing_value(self):
        result = make_result(1, "正文")
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[result]), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: ""}), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(["index", "问题"])

        self.assertEqual(code, 1)
        message = error.getvalue()
        self.assertIn(DEFAULT_ANSWER_API_KEY_ENV, message)
        self.assertIn("环境变量", message)

    def test_prints_answer_citations_and_evidence(self):
        result = make_result(1, "加班费按三档支付。")
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[result]), \
             patch("src.rag.cli.answering.create_llm", return_value=FakeModel("按三档支付 [1]。")), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: "test-key"}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["index", "加班费怎么算？"])

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("回答:\n按三档支付 [1]。", text)
        self.assertIn("引用检查: 回答引用了 [1]", text)
        self.assertIn("证据 1（已被引用）", text)
        self.assertIn("加班费按三档支付。", text)
        self.assertIn("物理页码: 291–292", text)
        self.assertIn("来源状态: 可回到当前 PDF 核对", text)
        self.assertIn("发送到上述地址", text)

    def test_warns_about_invalid_citation(self):
        result = make_result(1, "正文")
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[result]), \
             patch("src.rag.cli.answering.create_llm", return_value=FakeModel("结论 [9]。")), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: "test-key"}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["index", "问题"])

        self.assertEqual(code, 0)
        self.assertIn("不存在的证据编号 [9]", output.getvalue())

    def test_warns_when_cited_evidence_cannot_be_verified(self):
        result = make_result(1, "正文", provenance_complete=False)
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[result]), \
             patch("src.rag.cli.answering.create_llm", return_value=FakeModel("结论 [1]。")), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: "test-key"}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["index", "问题"])

        self.assertEqual(code, 0)
        self.assertIn("来源检查: 引用的证据 [1] 无法回到当前文件核对", output.getvalue())
        self.assertIn("来源状态: 无法完成当前文件核对；来源文件内容已变化", output.getvalue())

    def test_generation_failure_still_shows_retrieved_evidence(self):
        result = make_result(1, "加班费按三档支付。")
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[result]), \
             patch("src.rag.cli.answering.create_llm", side_effect=ValueError("鉴权失败")), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: "test-key"}), \
             contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()) as error:
            code = main(["index", "问题"])

        self.assertEqual(code, 1)
        self.assertIn("鉴权失败", error.getvalue())
        self.assertIn("已取得但未生成回答", output.getvalue())
        self.assertIn("加班费按三档支付。", output.getvalue())

    def test_empty_results_reports_insufficient_without_key(self):
        with patch("src.rag.cli.answering.retrieve_and_rerank", return_value=[]), \
             patch.dict(os.environ, {DEFAULT_ANSWER_API_KEY_ENV: ""}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["index", "问题"])

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("资料不足", text)
        self.assertIn("未调用生成模型", text)


if __name__ == "__main__":
    unittest.main()
