import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from src.rag.indexing.model import snapshot_identity
from src.rag.indexing.store import build_index
from src.rag.retrieval.search import RetrievalResult, retrieve, rerank, retrieve_and_rerank
from src.rag.retrieval.search import _source_status


class FakeLocalEmbedding(MockEmbedding):
    """测试用本地嵌入模型，维度与构建索引保持一致。"""

    def __init__(self, *args, **kwargs):
        super().__init__(embed_dim=3)


def make_snapshot(root: Path, *, marker: str = "same") -> Path:
    snapshot = root / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")
    return snapshot


def make_index(root: Path, *, nodes: list[TextNode] | None = None) -> tuple[Path, Path]:
    model_path = make_snapshot(root)
    identity = snapshot_identity(model_path)
    index_path = root / "index"
    if nodes is None:
        source = root / "source.pdf"
        source.write_bytes(b"pdf")
        nodes = [
            TextNode(text="alpha answer", metadata={"file_name": "source.pdf", "source_path": str(source), "source_id": "source", "page_numbers": [2]}),
            TextNode(text="beta answer", metadata={"file_name": "source.pdf", "source_path": str(source), "source_id": "source", "page_numbers": [3, 4]}),
        ]
    documents = []
    for source_path in sorted({str(node.metadata.get("source_path")) for node in nodes}):
        source = Path(source_path)
        if source.is_file():
            documents.append({"path": source_path, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    build_index(nodes, MockEmbedding(embed_dim=3), index_path, {
        **identity, "dimension": 3, "chunk_size": 128, "chunk_overlap": 32,
        "documents": documents,
    })
    return index_path, model_path


class RetrievalCoreTests(unittest.TestCase):
    @staticmethod
    def make_result(rank: int, text: str, score: float) -> RetrievalResult:
        return RetrievalResult(
            rank=rank,
            score=score,
            node_id=f"node-{rank}",
            text=text,
            file_name="source.pdf",
            source_path="/docs/source.pdf",
            page_numbers=(rank,),
            source_exists=True,
            provenance_complete=True,
        )

    def test_retrieve_returns_ranked_complete_source_traceable_results(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, model_path = make_index(Path(temp))
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding):
                results = retrieve(index_path, model_path, "alpha", top_k=2)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(isinstance(item, RetrievalResult) for item in results))
            self.assertEqual([item.rank for item in results], [1, 2])
            self.assertTrue(all(item.score is not None for item in results))
            self.assertEqual(len({item.node_id for item in results}), 2)
            self.assertEqual({item.text for item in results}, {"alpha answer", "beta answer"})
            self.assertEqual({item.page_numbers for item in results}, {(2,), (3, 4)})
            self.assertTrue(all(item.provenance_complete for item in results))

    def test_rejects_changed_model_snapshot_before_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path, _ = make_index(root)
            changed = make_snapshot(root / "changed", marker="changed")
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding):
                with self.assertRaisesRegex(ValueError, "不兼容"):
                    retrieve(index_path, changed, "question")

    def test_excludes_nodes_without_valid_pdf_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.pdf"
            source.write_bytes(b"pdf")
            nodes = [
                TextNode(text="valid", metadata={"file_name": "source.pdf", "source_path": str(source), "source_id": "source", "page_numbers": [1]}),
                TextNode(text="missing page", metadata={"file_name": "source.pdf", "source_path": str(source), "source_id": "source"}),
                TextNode(text="bad page", metadata={"file_name": "source.pdf", "source_path": str(source), "source_id": "source", "page_numbers": [True]}),
            ]
            index_path, model_path = make_index(root, nodes=nodes)
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding):
                results = retrieve(index_path, model_path, "question", top_k=5)
            self.assertEqual([item.text for item in results], ["valid"])

    def test_validates_top_k_and_question(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, model_path = make_index(Path(temp))
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding):
                for value in (0, -1):
                    with self.subTest(top_k=value), self.assertRaisesRegex(ValueError, "Top-K"):
                        retrieve(index_path, model_path, "question", top_k=value)
                with self.assertRaisesRegex(ValueError, "问题"):
                    retrieve(index_path, model_path, "  ")
                self.assertLessEqual(len(retrieve(index_path, model_path, "question", top_k=99)), 2)

    def test_reloads_persisted_text_without_reading_source_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path, model_path = make_index(root)
            (root / "source.pdf").unlink()
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding), patch("src.rag.retrieval.search.load_pdf", side_effect=AssertionError("must not parse PDF"), create=True):
                results = retrieve(index_path, model_path, "question", top_k=1)
            self.assertIn(results[0].text, {"alpha answer", "beta answer"})
            self.assertFalse(results[0].source_exists)
            self.assertFalse(results[0].provenance_complete)

    def test_hashes_same_source_once_for_multiple_dense_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path, model_path = make_index(Path(temp))
            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding), patch(
                "src.rag.retrieval.search._source_status", wraps=_source_status
            ) as source_status:
                results = retrieve(index_path, model_path, "question", top_k=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(source_status.call_count, 1)

    def test_marks_changed_source_as_not_verifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_path, model_path = make_index(root)
            source = root / "source.pdf"
            manifest_path = index_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["documents"] = [{"path": str(source), "sha256": hashlib.sha256(b"pdf").hexdigest()}]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source.write_bytes(b"updated pdf")

            with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", FakeLocalEmbedding):
                results = retrieve(index_path, model_path, "question", top_k=1)

            self.assertTrue(results[0].source_exists)
            self.assertFalse(results[0].provenance_complete)
            self.assertIn("内容已变化", results[0].provenance_warning or "")

    def test_rerank_orders_candidates_and_preserves_dense_metadata(self):
        candidates = [
            self.make_result(1, "候选一", 0.91),
            self.make_result(2, "候选二", 0.89),
            self.make_result(3, "候选三", 0.87),
        ]

        class FakeCrossEncoder:
            def predict(self, pairs):
                self.pairs = pairs
                return [0.1, 0.9, 0.2]

        fake_model = FakeCrossEncoder()
        with tempfile.TemporaryDirectory() as temp:
            reranker_path = Path(temp) / "reranker"
            reranker_path.mkdir()
            with patch("src.rag.retrieval.search._load_cross_encoder", return_value=fake_model):
                results = rerank("问题", candidates, reranker_path, top_k=2)

        self.assertEqual([item.node_id for item in results], ["node-2", "node-3"])
        self.assertEqual([item.rank for item in results], [1, 2])
        self.assertEqual([item.original_rank for item in results], [2, 3])
        self.assertEqual([item.score for item in results], [0.89, 0.87])
        self.assertEqual([item.rerank_score for item in results], [0.9, 0.2])
        self.assertEqual(fake_model.pairs, [("问题", "候选一"), ("问题", "候选二"), ("问题", "候选三")])

    def test_rerank_rejects_non_scalar_model_scores_with_readable_error(self):
        candidates = [self.make_result(1, "候选一", 0.9)]

        class MultiLabelCrossEncoder:
            def predict(self, pairs):
                return [[0.2, 0.8]]

        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "reranker"
            model_path.mkdir()
            with patch("src.rag.retrieval.search._load_cross_encoder", return_value=MultiLabelCrossEncoder()):
                with self.assertRaisesRegex(ValueError, "必须为每个候选返回一个标量分数"):
                    rerank("问题", candidates, model_path)

    def test_rerank_returns_available_candidates_when_fewer_than_top_k(self):
        candidates = [self.make_result(1, "候选一", 0.9), self.make_result(2, "候选二", 0.8)]

        class FakeCrossEncoder:
            def predict(self, pairs):
                return [0.7, 0.6]

        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "reranker"
            model_path.mkdir()
            with patch("src.rag.retrieval.search._load_cross_encoder", return_value=FakeCrossEncoder()):
                results = rerank("问题", candidates, model_path, top_k=5)

        self.assertEqual([result.node_id for result in results], ["node-1", "node-2"])

    def test_rerank_skips_model_load_when_no_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            reranker_path = Path(temp) / "reranker"
            reranker_path.mkdir()
            with patch("src.rag.retrieval.search._load_cross_encoder") as loader:
                self.assertEqual(rerank("问题", [], reranker_path), [])
            loader.assert_not_called()

    def test_cross_encoder_load_uses_local_safe_options(self):
        from src.rag.retrieval.search import _load_cross_encoder

        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "reranker"
            model_path.mkdir()
            with patch("sentence_transformers.CrossEncoder", return_value=object()) as constructor:
                result = _load_cross_encoder(model_path)

        self.assertIsNotNone(result)
        constructor.assert_called_once_with(
            str(model_path.resolve()),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )

    def test_retrieve_and_rerank_uses_twenty_candidates_then_returns_five(self):
        candidates = [self.make_result(index, f"候选{index}", 1.0 - index / 100) for index in range(1, 21)]
        with tempfile.TemporaryDirectory() as temp:
            reranker_path = Path(temp) / "reranker"
            reranker_path.mkdir()
            with patch("src.rag.retrieval.search.retrieve", return_value=candidates) as retrieve_mock, patch(
                "src.rag.retrieval.search.rerank", return_value=candidates[:5]
            ) as rerank_mock:
                results = retrieve_and_rerank(
                    Path("/tmp/index"), Path("/tmp/embed"), reranker_path, "问题"
                )

            retrieve_mock.assert_called_once_with(Path("/tmp/index"), Path("/tmp/embed"), "问题", top_k=20)
            rerank_mock.assert_called_once_with("问题", candidates, reranker_path.resolve(), top_k=5)
            self.assertEqual(len(results), 5)

    def test_retrieve_and_rerank_accepts_fewer_candidates_than_requested(self):
        candidates = [self.make_result(1, "候选一", 0.9), self.make_result(2, "候选二", 0.8)]
        with tempfile.TemporaryDirectory() as temp:
            reranker_path = Path(temp) / "reranker"
            reranker_path.mkdir()
            with patch("src.rag.retrieval.search.retrieve", return_value=candidates) as retrieve_mock, patch(
                "src.rag.retrieval.search.rerank", return_value=candidates[:1]
            ) as rerank_mock:
                results = retrieve_and_rerank(
                    Path("/tmp/index"), Path("/tmp/embed"), reranker_path, "问题", candidate_k=20, top_k=5
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(rerank_mock.call_args.args[1]), 2)
        retrieve_mock.assert_called_once_with(Path("/tmp/index"), Path("/tmp/embed"), "问题", top_k=20)

    def test_rerank_rejects_invalid_limits_and_missing_model(self):
        candidates = [self.make_result(1, "候选", 0.9)]
        for value in (0, -1, True):
            with self.subTest(top_k=value), self.assertRaisesRegex(ValueError, "Top-K"):
                rerank("问题", candidates, Path("/tmp/reranker"), top_k=value)
        with self.assertRaisesRegex(ValueError, "候选"):
            retrieve_and_rerank(Path("/tmp/index"), Path("/tmp/embed"), Path("/tmp/reranker"), "问题", candidate_k=0)
        with self.assertRaisesRegex(ValueError, "不能大于"):
            retrieve_and_rerank(Path("/tmp/index"), Path("/tmp/embed"), Path("/tmp/reranker"), "问题", candidate_k=3, top_k=5)
        with self.assertRaisesRegex(ValueError, "重排模型不存在"):
            rerank("问题", candidates, Path("/tmp/missing-reranker"), top_k=1)
        with self.assertRaisesRegex(ValueError, "重排模型不存在"):
            rerank("问题", [], Path("/tmp/missing-reranker"), top_k=1)
        with tempfile.TemporaryDirectory() as temp:
            reranker_path = Path(temp) / "invalid-reranker"
            reranker_path.mkdir()
            with patch("src.rag.retrieval.search._load_cross_encoder", side_effect=ValueError("模型无法加载")):
                with self.assertRaisesRegex(ValueError, "模型无法加载"):
                    rerank("问题", candidates, reranker_path, top_k=1)


class RetrievalCliTests(unittest.TestCase):
    def test_search_uses_configured_embedding_path_when_not_overridden(self):
        from src.rag.cli.retrieval import main
        from src.rag.config.common import DEFAULT_EMBEDDING_MODEL_PATH

        with patch("src.rag.cli.retrieval.retrieve", return_value=[]) as retrieve_mock, contextlib.redirect_stdout(io.StringIO()):
            code = main(["search", "/tmp/index", "问题", "--dense-only"])

        self.assertEqual(code, 0)
        self.assertEqual(retrieve_mock.call_args.args[1], DEFAULT_EMBEDDING_MODEL_PATH)

    def test_search_command_displays_rerank_metadata(self):
        from src.rag.cli.retrieval import main
        result = RetrievalResult(
            rank=1, score=0.675, node_id="node-reranked", text="重排后的完整原文",
            file_name="source.pdf", source_path="/docs/source.pdf", page_numbers=(8,),
            source_exists=True, provenance_complete=True, original_rank=7, rerank_score=0.932,
        )
        with patch("src.rag.cli.retrieval.retrieve_and_rerank", return_value=[result]) as rerank_mock, contextlib.redirect_stdout(io.StringIO()) as output:
            code = main([
                "search", "/tmp/index", "问题内容", "--model-path", "/tmp/embed",
                "--rerank-model-path", "/tmp/reranker", "--candidate-k", "20", "--top-k", "5",
            ])
        self.assertEqual(code, 0)
        rerank_mock.assert_called_once()
        text = output.getvalue()
        for expected in ("检索模式: dense top 20 → rerank top 5", "原始排名: 7", "新排名: 1", "dense 相似度分数: 0.675000", "rerank 分数: 0.932000", "重排后的完整原文"):
            self.assertIn(expected, text)

    def test_search_command_reports_rerank_error_without_fallback(self):
        from src.rag.cli.retrieval import main
        with patch("src.rag.cli.retrieval.retrieve_and_rerank", side_effect=ValueError("重排模型不存在")) as rerank_mock, patch(
            "src.rag.cli.retrieval.retrieve", side_effect=AssertionError("must not fallback")
        ), contextlib.redirect_stderr(io.StringIO()) as output:
            code = main([
                "search", "/tmp/index", "问题", "--model-path", "/tmp/embed",
                "--rerank-model-path", "/tmp/reranker",
            ])
        self.assertEqual(code, 1)
        rerank_mock.assert_called_once()
        self.assertIn("检索失败", output.getvalue())
        self.assertIn("重排模型不存在", output.getvalue())

    def test_search_command_displays_rank_score_text_and_source(self):
        from src.rag.cli.retrieval import main
        result = RetrievalResult(
            rank=1, score=0.875, node_id="node-1", text="完整的候选原文",
            file_name="source.pdf", source_path="/docs/source.pdf", page_numbers=(4, 5),
            source_exists=True, provenance_complete=True,
        )
        with patch("src.rag.cli.retrieval.retrieve", return_value=[result]), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "问题内容", "--model-path", "/tmp/model", "--top-k", "3", "--dense-only"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        for expected in ("问题内容", "排名: 1", "0.875", "node-1", "source.pdf", "/docs/source.pdf", "4–5", "完整的候选原文", "相似度分数越高"):
            self.assertIn(expected, text)

    def test_search_command_reports_no_results(self):
        from src.rag.cli.retrieval import main
        with patch("src.rag.cli.retrieval.retrieve", return_value=[]), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "没有答案的问题", "--model-path", "/tmp/model", "--dense-only"])
        self.assertEqual(code, 0)
        self.assertIn("未找到可交付的检索结果", output.getvalue())

    def test_search_command_reports_readable_error(self):
        from src.rag.cli.retrieval import main
        with patch("src.rag.cli.retrieval.retrieve", side_effect=ValueError("索引配置不兼容")), contextlib.redirect_stderr(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "问题", "--model-path", "/tmp/model", "--dense-only"])
        self.assertEqual(code, 1)
        self.assertIn("检索失败", output.getvalue())
        self.assertIn("索引配置不兼容", output.getvalue())


if __name__ == "__main__":
    unittest.main()
