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
from src.rag.retrieval.search import RetrievalResult, retrieve


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


if __name__ == "__main__":
    unittest.main()

class RetrievalCliTests(unittest.TestCase):
    def test_search_command_displays_rank_score_text_and_source(self):
        from src.rag.cli.retrieval import main
        result = RetrievalResult(
            rank=1, score=0.875, node_id="node-1", text="完整的候选原文",
            file_name="source.pdf", source_path="/docs/source.pdf", page_numbers=(4, 5),
            source_exists=True, provenance_complete=True,
        )
        with patch("src.rag.cli.retrieval.retrieve", return_value=[result]), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "问题内容", "--model-path", "/tmp/model", "--top-k", "3"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        for expected in ("问题内容", "排名: 1", "0.875", "node-1", "source.pdf", "/docs/source.pdf", "4–5", "完整的候选原文", "相似度分数越高"):
            self.assertIn(expected, text)

    def test_search_command_reports_no_results(self):
        from src.rag.cli.retrieval import main
        with patch("src.rag.cli.retrieval.retrieve", return_value=[]), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "没有答案的问题", "--model-path", "/tmp/model"])
        self.assertEqual(code, 0)
        self.assertIn("未找到可交付的检索结果", output.getvalue())

    def test_search_command_reports_readable_error(self):
        from src.rag.cli.retrieval import main
        with patch("src.rag.cli.retrieval.retrieve", side_effect=ValueError("索引配置不兼容")), contextlib.redirect_stderr(io.StringIO()) as output:
            code = main(["search", "/tmp/index", "问题", "--model-path", "/tmp/model"])
        self.assertEqual(code, 1)
        self.assertIn("检索失败", output.getvalue())
        self.assertIn("索引配置不兼容", output.getvalue())
