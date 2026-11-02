"""局域网接口的参数校验、鉴权、序列化和错误映射测试。

这些测试不加载真实 embedding/重排模型：索引清单用临时目录里的最小 manifest
构造，检索结果和生成模型通过打桩替换。要验证的是 HTTP 层的契约——状态码、
字段、来源信息和错误提示，而不是检索质量。
"""

import json
import os
import tempfile
import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from src.rag.answering.generate import INSUFFICIENT_EVIDENCE_ANSWER
from src.rag.api.app import ApiSettings, create_app
from src.rag.config.api import API_TOKEN_ENV, MAX_CANDIDATE_K, MAX_QUESTION_CHARS, MAX_TOP_K
from src.rag.retrieval.search import RetrievalResult


def setUpModule():
    """把接口测试的日志写入临时文件，避免污染用户的运行日志。"""
    temp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(temp.cleanup)
    env = patch.dict(os.environ, {"RAG_LOG_FILE": str(Path(temp.name) / "test.log"), "RAG_LOG_LEVEL": "INFO"})
    env.start()
    unittest.addModuleCleanup(env.stop)


def make_result(rank: int = 1, *, complete: bool = True) -> RetrievalResult:
    """构造一条带来源的检索结果，来源状态可切换以检查警告文案。"""
    return RetrievalResult(
        rank=rank,
        score=0.9,
        node_id=f"node-{rank}",
        text="原文片段",
        file_name="source.pdf",
        source_path="/docs/source.pdf",
        page_numbers=(rank, rank + 1),
        source_exists=complete,
        provenance_complete=complete,
        provenance_warning=None if complete else "来源文件不存在或已移动: /docs/source.pdf",
    )


class FakeChatModel:
    """只实现 ``chat`` 的最小生成模型，返回带 ``[1]`` 引用的固定回答。"""

    def __init__(self, content: str = "答案 [1]") -> None:
        self.content = content

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.content))


class FakeStreamModel(FakeChatModel):
    """按块产出文本的流式模型；可指定结束原因模拟被截断的输出。"""

    def __init__(self, chunks: Sequence[str] = ("答案", " [1]"), finish_reason: str | None = None) -> None:
        super().__init__()
        self.chunks = chunks
        self.finish_reason = finish_reason

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[ChatResponse]:
        for text in self.chunks:
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                delta=text,
                raw={"choices": [{"finish_reason": None}]},
            )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
            delta="",
            raw={"choices": [{"finish_reason": self.finish_reason}]},
        )


def parse_sse(text: str) -> list[tuple[str, Any]]:
    """把 SSE 响应拆成 ``(事件名, 数据)`` 列表，供断言事件顺序和内容。"""
    events: list[tuple[str, Any]] = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


class ApiTestCase(unittest.TestCase):
    """准备一个可用的临时索引目录，并按需替换检索和生成。"""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.index = root / "index"
        self.index.mkdir()
        (self.index / "manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "chunk_count": 2,
                    "model": "BAAI/bge-small-zh-v1.5",
                    "version": "1.0",
                    "dimension": 3,
                }
            ),
            encoding="utf-8",
        )
        self.settings = ApiSettings(
            index=self.index,
            embed_model=root / "embed",
            rerank_model=root / "rerank",
            answer_model="test-model",
            answer_api_base="https://example.invalid/v1",
            answer_api_key_env="TEST_API_KEY",
        )

    def client(self) -> TestClient:
        """创建测试客户端；令牌默认显式清空，避免受本机环境影响。"""
        env = patch.dict(os.environ, {API_TOKEN_ENV: ""})
        env.start()
        self.addCleanup(env.stop)
        return TestClient(create_app(self.settings))


class HealthTests(ApiTestCase):
    def test_health_reports_index_summary(self) -> None:
        response = self.client().get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["chunk_count"], 2)
        self.assertEqual(body["model"], "BAAI/bge-small-zh-v1.5")
        self.assertFalse(body["auth_required"])
        self.assertFalse(body["rerank_model_available"])

    def test_health_reports_unavailable_index_without_failing(self) -> None:
        self.settings = ApiSettings(
            index=self.index / "missing",
            embed_model=self.settings.embed_model,
            rerank_model=self.settings.rerank_model,
            answer_model=self.settings.answer_model,
            answer_api_base=self.settings.answer_api_base,
            answer_api_key_env=self.settings.answer_api_key_env,
        )
        response = self.client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertIn("索引不存在", response.json()["error"])

    def test_health_does_not_require_token(self) -> None:
        with patch.dict(os.environ, {API_TOKEN_ENV: "secret"}):
            response = TestClient(create_app(self.settings)).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["auth_required"])


class SearchTests(ApiTestCase):
    def test_search_serializes_provenance(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]):
            response = self.client().post("/search", json={"question": "问题"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "dense+rerank")
        item = body["results"][0]
        self.assertEqual(item["node_id"], "node-1")
        self.assertEqual(item["page_numbers"], [1, 2])
        self.assertEqual(item["page_label"], "1–2")
        self.assertEqual(item["source_status"], "可回到当前 PDF 核对")

    def test_search_reports_incomplete_provenance(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1, complete=False)]):
            response = self.client().post("/search", json={"question": "问题"})
        item = response.json()["results"][0]
        self.assertFalse(item["provenance_complete"])
        self.assertIn("无法完成当前文件核对", item["source_status"])

    def test_dense_only_uses_dense_retriever(self) -> None:
        with patch("src.rag.api.app.retrieve", return_value=[make_result(1)]) as dense, patch(
            "src.rag.api.app.retrieve_and_rerank"
        ) as reranked:
            response = self.client().post("/search", json={"question": "问题", "dense_only": True, "top_k": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "dense")
        dense.assert_called_once()
        reranked.assert_not_called()


class AskTests(ApiTestCase):
    def test_ask_returns_answer_and_evidence(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]), patch(
            "src.rag.api.app.resolve_api_key", return_value="key"
        ), patch("src.rag.api.app.create_llm", return_value=FakeChatModel()):
            response = self.client().post("/ask", json={"question": "问题"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        body = response.json()
        self.assertEqual(body["answer"], "答案 [1]")
        self.assertEqual(body["citations"], [1])
        self.assertEqual(body["invalid_citations"], [])
        self.assertTrue(body["llm_called"])
        self.assertEqual(body["evidence"][0]["page_label"], "1–2")

    def test_ask_without_evidence_does_not_need_key(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[]), patch(
            "src.rag.api.app.resolve_api_key"
        ) as resolve:
            response = self.client().post("/ask", json={"question": "问题"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["llm_called"])
        self.assertEqual(response.json()["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        resolve.assert_not_called()

    def test_ask_missing_key_is_503(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]), patch(
            "src.rag.api.app.resolve_api_key", side_effect=ValueError("未设置环境变量 TEST_API_KEY")
        ):
            response = self.client().post("/ask", json={"question": "问题"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("未设置环境变量", response.json()["detail"])

    def test_ask_generation_failure_is_502(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]), patch(
            "src.rag.api.app.resolve_api_key", return_value="key"
        ), patch("src.rag.api.app.create_llm", side_effect=RuntimeError("连接超时")):
            response = self.client().post("/ask", json={"question": "问题"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("生成回答失败", response.json()["detail"])


class ValidationTests(ApiTestCase):
    def post(self, payload: dict[str, object]) -> Any:
        return self.client().post("/search", json=payload)

    def test_empty_question_is_400(self) -> None:
        response = self.post({"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("问题不能为空", response.json()["detail"])

    def test_overlong_question_is_400(self) -> None:
        response = self.post({"question": "问" * (MAX_QUESTION_CHARS + 1)})
        self.assertEqual(response.status_code, 400)

    def test_top_k_out_of_range_is_400(self) -> None:
        self.assertEqual(self.post({"question": "问题", "top_k": 0}).status_code, 400)
        self.assertEqual(self.post({"question": "问题", "top_k": MAX_TOP_K + 1}).status_code, 400)

    def test_candidate_k_out_of_range_is_400(self) -> None:
        self.assertEqual(self.post({"question": "问题", "candidate_k": MAX_CANDIDATE_K + 1}).status_code, 400)

    def test_top_k_greater_than_candidate_k_is_400(self) -> None:
        response = self.post({"question": "问题", "top_k": 5, "candidate_k": 2})
        self.assertEqual(response.status_code, 400)
        self.assertIn("candidate_k", response.json()["detail"])

    def test_missing_question_field_is_400(self) -> None:
        response = self.post({"top_k": 3})
        self.assertEqual(response.status_code, 400)

    def test_missing_index_is_404(self) -> None:
        self.settings = ApiSettings(
            index=self.index / "missing",
            embed_model=self.settings.embed_model,
            rerank_model=self.settings.rerank_model,
            answer_model=self.settings.answer_model,
            answer_api_base=self.settings.answer_api_base,
            answer_api_key_env=self.settings.answer_api_key_env,
        )
        response = self.client().post("/search", json={"question": "问题"})
        self.assertEqual(response.status_code, 404)

    def test_model_failure_is_503(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", side_effect=ValueError("索引与模型不兼容")):
            response = self.client().post("/search", json={"question": "问题"})
        self.assertEqual(response.status_code, 503)


class StreamTests(ApiTestCase):
    """`/ask` 在 `stream=true` 时的事件顺序、增量和失败上报。"""

    def stream(self, model: object) -> Any:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]), patch(
            "src.rag.api.app.resolve_api_key", return_value="key"
        ), patch("src.rag.api.app.create_llm", return_value=model):
            return self.client().post("/ask", json={"question": "问题", "stream": True})

    def test_stream_emits_status_evidence_and_deltas(self) -> None:
        response = self.stream(FakeStreamModel())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = parse_sse(response.text)
        self.assertEqual(events[0][0], "status")
        self.assertEqual(events[0][1]["stage"], "retrieving")
        self.assertEqual(events[1][0], "evidence")
        self.assertEqual(events[1][1]["items"][0]["node_id"], "node-1")
        self.assertEqual(events[2][1]["stage"], "generating")
        text = "".join(data["text"] for name, data in events if name == "delta")
        self.assertEqual(text, "答案 [1]")
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1]["answer"], "答案 [1]")
        self.assertEqual(events[-1][1]["citations"], [1])
        self.assertEqual(events[-1][1]["evidence"][0]["page_label"], "1–2")

    def test_stream_without_evidence_skips_generation(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", return_value=[]), patch(
            "src.rag.api.app.resolve_api_key"
        ) as resolve:
            response = self.client().post("/ask", json={"question": "问题", "stream": True})
        events = parse_sse(response.text)
        self.assertEqual([name for name, _ in events], ["status", "evidence", "done"])
        self.assertFalse(events[-1][1]["llm_called"])
        self.assertEqual(events[-1][1]["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        resolve.assert_not_called()

    def test_stream_reports_truncated_output(self) -> None:
        events = parse_sse(self.stream(FakeStreamModel(finish_reason="length")).text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("max_tokens", events[-1][1]["detail"])

    def test_stream_reports_generation_failure(self) -> None:
        class BrokenModel(FakeStreamModel):
            def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[ChatResponse]:
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="半截"),
                    delta="半截",
                    raw={"choices": [{"finish_reason": None}]},
                )
                raise RuntimeError("连接断开")

        events = parse_sse(self.stream(BrokenModel()).text)
        self.assertEqual(events[-2][0], "delta")
        self.assertEqual(events[-1][0], "error")
        self.assertIn("生成回答失败", events[-1][1]["detail"])

    def test_stream_reports_retrieval_failure(self) -> None:
        with patch("src.rag.api.app.retrieve_and_rerank", side_effect=ValueError("索引与模型不兼容")):
            response = self.client().post("/ask", json={"question": "问题", "stream": True})
        events = parse_sse(response.text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("索引与模型不兼容", events[-1][1]["detail"])

    def test_stream_requires_token(self) -> None:
        with patch.dict(os.environ, {API_TOKEN_ENV: "secret"}):
            response = TestClient(create_app(self.settings)).post("/ask", json={"question": "问题", "stream": True})
        self.assertEqual(response.status_code, 401)

    def test_stream_validates_before_sending(self) -> None:
        response = self.client().post("/ask", json={"question": "  ", "stream": True})
        self.assertEqual(response.status_code, 400)


class AuthTests(ApiTestCase):
    def test_request_without_token_is_401(self) -> None:
        with patch.dict(os.environ, {API_TOKEN_ENV: "secret"}):
            response = TestClient(create_app(self.settings)).post("/search", json={"question": "问题"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_wrong_token_is_401(self) -> None:
        with patch.dict(os.environ, {API_TOKEN_ENV: "secret"}):
            response = TestClient(create_app(self.settings)).post(
                "/search", json={"question": "问题"}, headers={"Authorization": "Bearer wrong"}
            )
        self.assertEqual(response.status_code, 401)

    def test_correct_token_passes(self) -> None:
        with patch.dict(os.environ, {API_TOKEN_ENV: "secret"}), patch(
            "src.rag.api.app.retrieve_and_rerank", return_value=[make_result(1)]
        ):
            response = TestClient(create_app(self.settings)).post(
                "/search", json={"question": "问题"}, headers={"Authorization": "Bearer secret"}
            )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
