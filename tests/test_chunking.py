import contextlib
import io
import unittest
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from src.rag.chunking import chunk_documents
from src.rag.chunking_cli import main


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_page_sources_and_show_adjacent_overlap(self):
        text = " ".join(f"Sentence {i} has several useful words." for i in range(20))
        doc = Document(text=text, metadata={"file_name": "a.pdf", "source_path": "/a.pdf", "source_id": "a", "page_number": 2, "source_location": "物理页码 2"})
        nodes = chunk_documents([doc], chunk_size=30, chunk_overlap=10)
        self.assertGreater(len(nodes), 1)
        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        self.assertTrue(all(text.strip() for text in texts))
        self.assertTrue(all(n.metadata["page_number"] == 2 and n.metadata["source_path"] == "/a.pdf" for n in nodes))
        self.assertTrue(any(texts[0].endswith(texts[1][:i]) for i in range(10, len(texts[1]) + 1)))

    def test_non_pdf_location_is_preserved_without_page_number(self):
        doc = Document(text="A paragraph.", metadata={"file_name": "a.txt", "source_path": "/a.txt", "source_id": "a", "source_location": "第 2 章，第 3 段"})
        nodes = chunk_documents([doc])
        self.assertEqual(nodes[0].metadata["source_location"], "第 2 章，第 3 段")
        self.assertNotIn("page_number", nodes[0].metadata)

    def test_missing_location_is_rejected(self):
        doc = Document(text="A paragraph.", metadata={"file_name": "a.txt", "source_path": "/a.txt", "source_id": "a"})
        with self.assertRaisesRegex(ValueError, "来源"):
            _ = chunk_documents([doc])

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
