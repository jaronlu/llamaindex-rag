import tempfile
import unittest
from pathlib import Path

from llama_index.core import StorageContext
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from src.rag.indexing.store import build_index, read_node


class IndexingTests(unittest.TestCase):
    def test_duplicate_node_ids_are_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "index"
            nodes = [TextNode(id_="same", text="first"), TextNode(id_="same", text="second")]
            with self.assertRaisesRegex(ValueError, "重复.*ID"):
                build_index(nodes, MockEmbedding(embed_dim=8), location, {})
            self.assertFalse(location.exists())

    def test_persist_and_read_node_without_source_file(self):
        with tempfile.TemporaryDirectory() as temp:
            node = TextNode(text="原文内容", metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [2]})
            location = Path(temp) / "index"
            build_index([node], MockEmbedding(embed_dim=8), location, {"model": "mock", "version": "1", "chunk_size": 128, "chunk_overlap": 32, "documents": []})
            restored = read_node(location, node.node_id)
            self.assertEqual(restored.get_content(), "原文内容")
            self.assertEqual(restored.metadata["page_numbers"], [2])
            self.assertEqual(restored.metadata["source_path"], "/a.pdf")

    def test_existing_index_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "index"
            location.mkdir()
            with self.assertRaisesRegex(ValueError, "已存在"):
                build_index([TextNode(text="text")], MockEmbedding(embed_dim=8), location, {})

    def test_failure_does_not_publish_partial_index(self):
        class FailingEmbedding(MockEmbedding):
            def _get_text_embedding(self, text):
                raise ValueError("model failed")
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "index"
            with self.assertRaisesRegex(ValueError, "model failed"):
                build_index([TextNode(text="text")], FailingEmbedding(embed_dim=8), location, {})
            self.assertFalse(location.exists())


class IndexConfigurationTests(unittest.TestCase):
    def test_rejects_incompatible_configuration(self):
        from src.rag.indexing.store import check_compatibility
        with self.assertRaisesRegex(ValueError, "不兼容"):
            check_compatibility({"model": "a", "version": "1", "dimension": 8, "chunk_size": 128, "chunk_overlap": 32}, {"model": "a", "version": "2", "dimension": 8, "chunk_size": 128, "chunk_overlap": 32})

class IndexCliTests(unittest.TestCase):
    def test_preflight_failure_reports_unknown_total_and_unpublished(self):
        import contextlib
        import io
        from src.rag.cli.indexing import main
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index"
            with contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main(["build", str(index), str(Path(temp) / "missing.pdf"), "--model-path", temp]), 1)
            for text in ("向量化成功 0", "向量化失败 0", "未处理", "未知", "未发布"):
                self.assertIn(text, output.getvalue())
            self.assertFalse(index.exists())

    def test_page_failure_includes_underlying_error(self):
        import contextlib
        import io
        from unittest.mock import patch
        from src.rag.cli.indexing import main
        from src.rag.ingestion.pdf import LoadResult, PageResult
        from src.rag.config.ingestion import PageStatus
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp) / "source.pdf"
            pdf.write_bytes(b"pdf")
            loaded = LoadResult(pdf, 1, [PageResult(1, PageStatus.FAILED, error="broken page")])
            with patch("src.rag.cli.indexing.load_pdf", return_value=loaded), contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main(["build", str(Path(temp) / "index"), str(pdf), "--model-path", temp]), 1)
            self.assertIn("broken page", output.getvalue())

    def test_read_command_displays_stored_node(self):
        import contextlib
        import io
        from src.rag.cli.indexing import main
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "index"
            node = TextNode(text="可核对的原文", metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [2]})
            build_index([node], MockEmbedding(embed_dim=8), location, {})
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = main(["read", str(location), node.node_id])
            self.assertEqual(result, 0)
            self.assertIn("可核对的原文", output.getvalue())
            self.assertIn("/a.pdf", output.getvalue())

    def test_read_warns_when_source_no_longer_exists(self):
        import contextlib
        import io
        from src.rag.cli.indexing import main
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "index"
            node = TextNode(text="正文", metadata={"file_name": "gone.pdf", "source_path": "/missing/gone.pdf", "source_id": "a", "page_numbers": [1]})
            build_index([node], MockEmbedding(embed_dim=8), location, {"documents": [{"path": "/missing/gone.pdf", "sha256": "abc"}]})
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["read", str(location), node.node_id]), 0)
            self.assertIn("来源文件不存在", output.getvalue())

class ModelIdentityTests(unittest.TestCase):
    def test_same_named_snapshots_with_different_weights_are_incompatible(self):
        from src.rag.indexing.model import snapshot_identity
        from src.rag.indexing.store import check_compatibility
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshots = [root / name / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / "revision" for name in ("first", "second")]
            for path, content in zip(snapshots, (b"weights one", b"weights two")):
                path.mkdir(parents=True)
                (path / "model.safetensors").write_bytes(content)
            first, second = [snapshot_identity(path) for path in snapshots]
            self.assertNotEqual(first["model_sha256"], second["model_sha256"])
            config = {"dimension": 512, "chunk_size": 128, "chunk_overlap": 32}
            with self.assertRaisesRegex(ValueError, "不兼容"):
                check_compatibility({**first, **config}, {**second, **config})

    def test_rejects_unidentifiable_snapshot_path(self):
        from src.rag.indexing.model import snapshot_identity
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "快照"):
                snapshot_identity(Path(temp))

class CoverageTests(unittest.TestCase):
    def test_empty_pages_are_recorded_and_reported_after_reload(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from src.rag.cli.indexing import main
        from src.rag.ingestion.pdf import LoadResult, PageResult
        from src.rag.config.ingestion import PageStatus
        from llama_index.core import Document
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = (root / "source.pdf").resolve()
            pdf.write_bytes(b"pdf")
            model_path = root / "models--a--b" / "snapshots" / "v1"
            model_path.mkdir(parents=True)
            (model_path / "config.json").write_text('{}')
            doc = Document(text="正文", metadata={"file_name": "source.pdf", "source_path": str(pdf), "source_id": "a", "page_numbers": [1]})
            loaded = LoadResult(pdf, 2, [PageResult(1, PageStatus.SUCCESS, doc), PageResult(2, PageStatus.EMPTY)])
            class Tokenizer:
                def encode(self, text, **kwargs):
                    return [1]
            from pydantic import PrivateAttr
            class LocalEmbedding(MockEmbedding):
                max_length: int = 512
                _model: object = PrivateAttr(default_factory=lambda: type("LocalModel", (), {"tokenizer": Tokenizer(), "prompts": {"text": None}})())
            index = root / "index"
            with patch("src.rag.cli.indexing.load_pdf", return_value=loaded), patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=LocalEmbedding(embed_dim=8)), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["build", str(index), str(pdf), "--model-path", str(model_path)]), 0)
            manifest = json.loads((index / "manifest.json").read_text())
            self.assertEqual(manifest["documents"][0]["empty_pages"], [2])
            self.assertFalse(manifest["documents"][0]["text_coverage_complete"])
            node_id = next(iter(StorageContext.from_defaults(persist_dir=str(index)).docstore.docs))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["read", str(index), node_id]), 0)
            self.assertIn("空文本页", out.getvalue())

class FailureAccountingTests(unittest.TestCase):
    def test_embedding_failure_separates_failed_and_unprocessed(self):
        class FailingEmbedding(MockEmbedding):
            def _get_text_embedding(self, text):
                if text == "fail":
                    raise ValueError("embedding error")
                return super()._get_text_embedding(text)
        with tempfile.TemporaryDirectory() as temp:
            nodes = [TextNode(text=text) for text in ("ok", "fail", "later")]
            with self.assertRaises(ValueError) as caught:
                build_index(nodes, FailingEmbedding(embed_dim=8), Path(temp) / "index", {})
            message = str(caught.exception)
            for part in ("向量化成功 1", "向量化失败 1", "未处理 1", "未发布"):
                self.assertIn(part, message)

    def test_persist_failure_reports_no_published_index(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            with patch("llama_index.core.storage.storage_context.StorageContext.persist", side_effect=OSError("disk error")):
                with self.assertRaises(ValueError) as caught:
                    build_index([TextNode(text="ok")], MockEmbedding(embed_dim=8), Path(temp) / "index", {})
            message = str(caught.exception)
            for part in ("向量化成功 1", "向量化失败 0", "未处理 0", "索引发布失败", "未发布"):
                self.assertIn(part, message)
            self.assertFalse((Path(temp) / "index").exists())

class PublicationRaceTests(unittest.TestCase):
    def test_competing_empty_directory_is_not_replaced(self):
        from unittest.mock import patch
        import src.rag.indexing.store as store
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'index'
            original = store.publish_index
            def competing_publish(staging, destination):
                destination.mkdir()
                return original(staging, destination)
            with patch.object(store, 'publish_index', side_effect=competing_publish):
                with self.assertRaisesRegex(ValueError, '已存在'):
                    build_index([TextNode(text='text')], MockEmbedding(embed_dim=8), target, {})
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])

class IndexOverviewTests(unittest.TestCase):
    def test_overview_shows_documents_without_nodes_after_reload(self):
        import contextlib
        import io
        from src.rag.cli.indexing import main
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / 'index'
            docs = [
                {'path': '/nonempty.pdf', 'start': 1, 'end': 1, 'empty_pages': [], 'text_coverage_complete': True},
                {'path': '/empty.pdf', 'start': 1, 'end': 2, 'empty_pages': [1, 2], 'text_coverage_complete': False},
            ]
            node = TextNode(text='正文', metadata={'source_path': '/nonempty.pdf', 'page_numbers': [1]})
            build_index([node], MockEmbedding(embed_dim=8), index, {'documents': docs})
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['overview', str(index)]), 0)
            self.assertIn('/empty.pdf', output.getvalue())
            self.assertIn('[1, 2]', output.getvalue())
            self.assertIn('不完整', output.getvalue())


if __name__ == "__main__":
    unittest.main()
