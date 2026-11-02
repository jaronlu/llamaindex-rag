import tempfile
import os
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode

from src.rag.chunking.split import chunk_documents
from src.rag.chunking.offset_splitter import split_with_offsets, structure_block_starts
from src.rag.cli.chunking import main


def setUpModule():
    """把切片测试的日志写入临时文件，避免污染用户的运行日志。"""
    temp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(temp.cleanup)
    env = patch.dict(os.environ, {'RAG_LOG_FILE': str(Path(temp.name) / 'test.log'), 'RAG_LOG_LEVEL': 'INFO'})
    env.start()
    unittest.addModuleCleanup(env.stop)


class ChunkingTests(unittest.TestCase):
    def test_concurrent_splits_return_their_own_offsets(self):
        texts = ["  First sentence. Second sentence.", "Another sentence.  "]
        def split(text):
            return split_with_offsets(Document(text=text), 30, 5)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(split, texts))
        for text, (nodes, offsets) in zip(texts, results):
            self.assertEqual(len(nodes), len(offsets))
            self.assertTrue(nodes)
            for node, (start, end) in zip(nodes, offsets):
                self.assertEqual(text[start:end], node.get_content(metadata_mode=MetadataMode.NONE))

    def test_offset_splitter_matches_native_sentence_splitter_without_structure_blocks(self):
        # 没有结构块时，适配器应退化为原生 SentenceSplitter。这条测试用于在升级
        # LlamaIndex 后立即发现私有 _Split/_merge 契约漂移，而不是等切片结果悄悄变化。
        texts = [
            "First sentence. Second sentence. Third sentence.",
            " ".join(f"Sentence {index} has several words." for index in range(12)),
        ]
        for text in texts:
            self.assertEqual(structure_block_starts(text), [0, len(text)])
            for chunk_size, chunk_overlap in ((30, 5), (30, 0), (16, 8), (64, 16)):
                with self.subTest(chunk_size=chunk_size, chunk_overlap=chunk_overlap, text=text[:20]):
                    adapted, _ = split_with_offsets(Document(text=text), chunk_size, chunk_overlap)
                    native = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents([Document(text=text)])
                    self.assertEqual(
                        [node.get_content(metadata_mode=MetadataMode.NONE) for node in adapted],
                        [node.get_content(metadata_mode=MetadataMode.NONE) for node in native],
                    )

    def test_boolean_page_numbers_are_rejected(self):
        for value in (True, False):
            doc = Document(text="text", metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [value]})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "页码"):
                chunk_documents([doc])

    def test_consecutive_pages_form_cross_page_chunk_with_page_range(self):
        def page(number, text, source="a"):
            return Document(text=text, metadata={"file_name": source + ".pdf", "source_path": "/" + source + ".pdf", "source_id": source, "page_numbers": [number]})
        nodes = chunk_documents([page(2, "The reason is"), page(3, "the following conclusion.")], 30, 5)
        self.assertEqual(len(nodes), 1)
        self.assertIn("The reason is\nthe following conclusion.", nodes[0].get_content(metadata_mode=MetadataMode.NONE))
        self.assertEqual(nodes[0].metadata["page_numbers"], [2, 3])
        self.assertNotIn("page_number", nodes[0].metadata)

    def test_gaps_and_different_files_do_not_join(self):
        def page(number, source):
            return Document(text=f"{source} page {number}", metadata={"file_name": source + ".pdf", "source_path": "/" + source + ".pdf", "source_id": source, "page_numbers": [number]})
        nodes = chunk_documents([page(1, "a"), page(3, "a"), page(4, "b")], 30, 5)
        self.assertEqual(len(nodes), 3)
        self.assertEqual([n.metadata["page_numbers"] for n in nodes], [[1], [3], [4]])

    def test_chunks_keep_page_sources_and_show_adjacent_overlap(self):
        text = " ".join(f"Sentence {i} has several useful words." for i in range(20))
        doc = Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [2]})
        nodes = chunk_documents([doc], chunk_size=30, chunk_overlap=10)
        self.assertGreater(len(nodes), 1)
        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        self.assertTrue(all(text.strip() for text in texts))
        self.assertTrue(all(n.metadata["page_numbers"] == [2] and n.metadata["source_path"] == "/a.pdf" for n in nodes))
        self.assertTrue(any(texts[0].endswith(texts[1][:i]) for i in range(10, len(texts[1]) + 1)))

    def test_chunk_boundaries_prefer_structure_blocks(self):
        text = "\n".join([
            "1. 第一条目。",
            "• 成本：不花钱。",
            "• 说人话：按 1.5 倍给钱。",
            "2. 第二条目。",
            "• 来源：官方文件。",
        ])
        doc = Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [1]})
        nodes = chunk_documents([doc], 30, 5)
        starts = structure_block_starts(text)
        self.assertGreater(len(starts), 2)
        self.assertGreater(len(nodes), 1)
        # 每个片段都从结构块起点开始，不会落在条目或来源行的中间。
        for node in nodes:
            assert isinstance(node, TextNode)
            self.assertIn(node.start_char_idx, starts)

    def test_oversized_structure_block_falls_back_to_sentence_split(self):
        text = "1. 标题。\n• 说人话：" + "这是一句很长的说明。" * 20
        doc = Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [1]})
        nodes = chunk_documents([doc], 30, 5)
        self.assertGreater(len(nodes), 1)
        for node in nodes:
            assert isinstance(node, TextNode)
            start, end = node.start_char_idx, node.end_char_idx
            assert start is not None and end is not None
            self.assertEqual(text[start:end], node.get_content(metadata_mode=MetadataMode.NONE))

    def test_non_pdf_is_rejected(self):
        doc = Document(text="A paragraph.", metadata={"file_name": "a.txt", "source_path": "/a.txt", "source_id": "a"})
        with self.assertRaisesRegex(ValueError, "PDF"):
            chunk_documents([doc])

    def test_repeated_text_across_pages_has_correct_locations(self):
        text = "Same sentence. " * 14
        docs = [Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [i]}) for i in (1, 2)]
        nodes = chunk_documents(docs, 30, 0)
        self.assertEqual([node.metadata["page_numbers"] for node in nodes], [[1], [1], [2], [2]])

    def test_pdf_requires_page_numbers(self):
        metadata = {"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a"}
        with self.assertRaisesRegex(ValueError, "页码"):
            chunk_documents([Document(text="text", metadata=metadata)])

    def test_rejects_obsolete_source_location(self):
        doc = Document(text="text", metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [1], "source_location": "物理页码 99"})
        with self.assertRaisesRegex(ValueError, "source_location"):
            chunk_documents([doc])

    def test_repeated_text_with_overlap_has_accurate_offsets(self):
        text = "Same sentence. " * 14
        docs = [Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [i]}) for i in (1, 2)]
        combined = text + "\n" + text
        nodes = chunk_documents(docs, 30, 10)
        for node in nodes:
            assert isinstance(node, TextNode)
            start, end = node.start_char_idx, node.end_char_idx
            assert start is not None and end is not None
            self.assertEqual(combined[start:end], node.get_content(metadata_mode=MetadataMode.NONE))
            expected = ([1] if start < len(text) else []) + ([2] if end > len(text) + 1 else [])
            self.assertEqual(node.metadata["page_numbers"], expected)

    def test_invalid_configuration_and_missing_source_are_rejected(self):
        doc = Document(text="some text")
        with self.assertRaises(ValueError):
            _ = chunk_documents([doc], chunk_size=10, chunk_overlap=10)
        with self.assertRaises(ValueError):
            _ = chunk_documents([doc], chunk_size=30, chunk_overlap=5)

    def test_real_pdf_cli_displays_sources_parameters_and_chunks(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(["docs/pdf/HowToLiveBetter.pdf", "--start", "2", "--end", "3", "--chunk-size", "128", "--chunk-overlap", "32"])
        self.assertEqual(result, 0)
        for expected in ("128", "32", "物理页码", "片段", str(Path("docs/pdf/HowToLiveBetter.pdf").resolve())):
            self.assertIn(expected, output.getvalue())


if __name__ == "__main__":
    _ = unittest.main()
