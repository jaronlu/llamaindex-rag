import contextlib
import io
import unittest
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import MetadataMode, TextNode

from src.rag.chunking.split import chunk_documents
from src.rag.cli.chunking import main


class ChunkingTests(unittest.TestCase):
    def test_consecutive_pages_form_cross_page_chunk_with_page_range(self):
        def page(number, text, source="a"):
            return Document(text=text, metadata={"file_name": source + ".pdf", "source_path": "/" + source + ".pdf", "source_id": source, "page_numbers": [number], "source_location": f"物理页码 {number}"})
        nodes = chunk_documents([page(2, "The reason is"), page(3, "the following conclusion.")], 30, 5)
        self.assertEqual(len(nodes), 1)
        self.assertIn("The reason is\nthe following conclusion.", nodes[0].get_content(metadata_mode=MetadataMode.NONE))
        self.assertNotIn("source_location", nodes[0].metadata)
        self.assertEqual(nodes[0].metadata["page_numbers"], [2, 3])
        self.assertNotIn("page_number", nodes[0].metadata)

    def test_gaps_and_different_files_do_not_join(self):
        def page(number, source):
            return Document(text=f"{source} page {number}", metadata={"file_name": source + ".pdf", "source_path": "/" + source + ".pdf", "source_id": source, "page_numbers": [number], "source_location": f"物理页码 {number}"})
        nodes = chunk_documents([page(1, "a"), page(3, "a"), page(4, "b")], 30, 5)
        self.assertEqual(len(nodes), 3)
        self.assertEqual([n.metadata["page_numbers"] for n in nodes], [[1], [3], [4]])

    def test_chunks_keep_page_sources_and_show_adjacent_overlap(self):
        text = " ".join(f"Sentence {i} has several useful words." for i in range(20))
        doc = Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [2], "source_location": "物理页码 2"})
        nodes = chunk_documents([doc], chunk_size=30, chunk_overlap=10)
        self.assertGreater(len(nodes), 1)
        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        self.assertTrue(all(text.strip() for text in texts))
        self.assertTrue(all(n.metadata["page_numbers"] == [2] and n.metadata["source_path"] == "/a.pdf" for n in nodes))
        self.assertTrue(any(texts[0].endswith(texts[1][:i]) for i in range(10, len(texts[1]) + 1)))

    def test_non_pdf_location_is_preserved_without_page_numbers(self):
        doc = Document(text="A paragraph.", metadata={"file_name": "a.txt", "source_path": "/a.txt", "source_id": "a", "source_location": "第 2 章，第 3 段"})
        nodes = chunk_documents([doc])
        self.assertEqual(nodes[0].metadata["source_location"], "第 2 章，第 3 段")
        self.assertNotIn("page_numbers", nodes[0].metadata)

    def test_missing_location_is_rejected(self):
        doc = Document(text="A paragraph.", metadata={"file_name": "a.txt", "source_path": "/a.txt", "source_id": "a"})
        with self.assertRaisesRegex(ValueError, "来源"):
            _ = chunk_documents([doc])

    def test_repeated_text_across_pages_has_correct_locations(self):
        text = "Same sentence. " * 14
        docs = [Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_numbers": [i]}) for i in (1, 2)]
        nodes = chunk_documents(docs, 30, 0)
        self.assertEqual([node.metadata["page_numbers"] for node in nodes], [[1], [1], [2], [2]])

    def test_pdf_requires_page_numbers_and_does_not_keep_text_location(self):
        metadata: dict[str, object] = {"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "source_location": "物理页码 99"}
        with self.assertRaisesRegex(ValueError, "页码"):
            chunk_documents([Document(text="text", metadata=metadata)])
        metadata["page_numbers"] = [1]
        node = chunk_documents([Document(text="text", metadata=metadata)])[0]
        self.assertNotIn("source_location", node.metadata)

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
