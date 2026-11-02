import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from src.rag.config.indexing import HASH_READ_SIZE
from src.rag.ingestion.pdf import load_pdf
from src.rag.provenance import (
    document_record,
    document_records,
    file_sha256,
    format_pages,
    read_manifest,
    source_id_for,
    source_status,
)


class FileHashTests(unittest.TestCase):
    def test_matches_hashlib_across_multiple_read_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.bin"
            data = b"x" * (HASH_READ_SIZE + 1)
            path.write_bytes(data)
            self.assertEqual(file_sha256(path), hashlib.sha256(data).hexdigest())

    def test_empty_file_has_known_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.bin"
            path.write_bytes(b"")
            self.assertEqual(
                file_sha256(path),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            )


class SourceIdTests(unittest.TestCase):
    def test_same_resolved_path_is_stable_and_different_paths_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "document.pdf"
            variant = root / "." / "document.pdf"
            self.assertEqual(source_id_for(path), source_id_for(path))
            self.assertEqual(source_id_for(path), source_id_for(variant))
            self.assertNotEqual(source_id_for(path), source_id_for(root / "other.pdf"))

    def test_ingestion_source_id_matches_build_time_derivation(self):
        # manifest 记录和节点 metadata 由不同阶段生成；这里锁住两者使用同一规则，
        # 避免以后只改一处导致 source_id 静默漂移。
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.pdf"
            writer = PdfWriter()
            _ = writer.add_blank_page(width=100, height=100)
            _ = writer.write(path)
            result = load_pdf(path, 1, 1)
            document = result.pages[0].document
            assert document is not None
            self.assertEqual(
                document.metadata["source_id"],
                source_id_for(Path(document.metadata["source_path"])),
            )


class ReadManifestTests(unittest.TestCase):
    def test_rejects_missing_directory_and_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "索引不存在"):
                read_manifest(root / "missing")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "缺少清单文件"):
                read_manifest(empty)

    def test_rejects_unreadable_non_dict_and_incomplete_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "无法读取"):
                read_manifest(root)
            manifest.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不完整"):
                read_manifest(root)
            manifest.write_text(json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不完整"):
                read_manifest(root)
            # 非法 UTF-8 与 JSON 语法错误同属“内容不可用”，不能漏成原始编解码异常。
            manifest.write_bytes(b'{"complete": true, "x": "\xff\xfe"}')
            with self.assertRaisesRegex(ValueError, "无法读取"):
                read_manifest(root)

    def test_returns_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {"complete": True, "chunk_count": 2}
            (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(read_manifest(root), payload)


class DocumentRecordTests(unittest.TestCase):
    def test_skips_corrupt_records_and_non_list_documents(self):
        manifest = {"documents": ["not-a-record", {"path": "/a.pdf", "sha256": "abc"}]}
        self.assertEqual([item["path"] for item in document_records(manifest)], ["/a.pdf"])
        self.assertEqual(document_record("/a.pdf", manifest), {"path": "/a.pdf", "sha256": "abc"})
        self.assertIsNone(document_record("/missing.pdf", manifest))
        self.assertEqual(document_records({"documents": "broken"}), [])
        self.assertIsNone(document_record("/a.pdf", {}))


class FormatPagesTests(unittest.TestCase):
    def test_empty_single_and_contiguous_pages(self):
        self.assertEqual(format_pages(()), "无")
        self.assertEqual(format_pages((291,)), "291")
        self.assertEqual(format_pages((291, 292)), "291–292")
        # 未排序的连续页码仍应展示成升序区间。
        self.assertEqual(format_pages((292, 291)), "291–292")

    def test_non_contiguous_pages_are_listed_separately(self):
        self.assertEqual(format_pages((291, 295)), "291、295")
        self.assertEqual(format_pages((296, 291, 295)), "291、295、296")


class SourceStatusTests(unittest.TestCase):
    @staticmethod
    def manifest(path: Path, digest: str | None) -> dict:
        record = {"path": str(path)}
        if digest is not None:
            record["sha256"] = digest
        return {"documents": [record]}

    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gone.pdf"
            exists, complete, warning = source_status(str(path), self.manifest(path, "abc"))
            self.assertFalse(exists)
            self.assertFalse(complete)
            self.assertIn("来源文件不存在", warning or "")

    def test_missing_record_or_digest_marks_source_unverifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.pdf"
            path.write_bytes(b"pdf")
            manifests: tuple[dict, ...] = ({"documents": []}, self.manifest(path, None))
            for manifest in manifests:
                with self.subTest(manifest=manifest):
                    exists, complete, warning = source_status(str(path), manifest)
                    self.assertTrue(exists)
                    self.assertFalse(complete)
                    self.assertIn("缺少来源版本摘要", warning or "")

    def test_matching_digest_is_verifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.pdf"
            path.write_bytes(b"pdf")
            manifest = self.manifest(path, hashlib.sha256(b"pdf").hexdigest())
            self.assertEqual(source_status(str(path), manifest), (True, True, None))

    def test_changed_content_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.pdf"
            path.write_bytes(b"updated")
            manifest = self.manifest(path, hashlib.sha256(b"pdf").hexdigest())
            exists, complete, warning = source_status(str(path), manifest)
            self.assertTrue(exists)
            self.assertFalse(complete)
            self.assertIn("内容已变化", warning or "")

    def test_unreadable_file_is_reported_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.pdf"
            path.write_bytes(b"pdf")
            manifest = self.manifest(path, "abc")
            with patch("src.rag.provenance.verify.file_sha256", side_effect=OSError("denied")):
                exists, complete, warning = source_status(str(path), manifest)
            self.assertTrue(exists)
            self.assertFalse(complete)
            self.assertIn("无法读取来源文件", warning or "")


if __name__ == "__main__":
    unittest.main()
