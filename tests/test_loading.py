import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf._page import PageObject

from src.rag.ingestion import load_pdf
from src.rag import ingestion
from src.rag.cli import main


class LoadingTests(unittest.TestCase):
    def test_selected_pages_include_physical_page_and_empty_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            writer = PdfWriter()
            for _ in range(3):
                _ = writer.add_blank_page(width=100, height=100)
            _ = writer.write(path)
            result = load_pdf(path, 2, 3)
            self.assertEqual(result.total_pages, 3)
            self.assertEqual([page.number for page in result.pages], [2, 3])
            self.assertTrue(all(page.status == "empty" for page in result.pages))
            self.assertIs(result.pages[0].status, ingestion.PageStatus.EMPTY)
            document = result.pages[0].document
            assert document is not None
            self.assertEqual(document.metadata["page_number"], 2)
            self.assertEqual(document.metadata["source_path"], str(path.resolve()))

    def test_one_page_failure_does_not_hide_other_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            writer = PdfWriter()
            _ = writer.add_blank_page(width=100, height=100)
            _ = writer.add_blank_page(width=100, height=100)
            _ = writer.write(path)
            original = PageObject.extract_text
            count = 0
            def extract(page: PageObject) -> str:
                nonlocal count
                count += 1
                if count == 1:
                    raise ValueError("bad page")
                return original(page)
            with patch.object(PageObject, "extract_text", extract):
                result = load_pdf(path)
            self.assertEqual([page.status for page in result.pages], ["failed", "empty"])
            error = result.pages[0].error
            assert error is not None
            self.assertIn("bad page", error)

    def test_invalid_range_and_missing_file_are_reported(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertNotEqual(main(["missing.pdf"]), 0)
        self.assertIn("missing.pdf", error.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            writer = PdfWriter()
            _ = writer.add_blank_page(width=100, height=100)
            _ = writer.write(path)
            with self.assertRaises(ValueError):
                _ = load_pdf(path, 2, 3)

    def test_real_pdf_selected_page(self):
        path = Path("docs/pdf/HowToLiveBetter.pdf")
        result = load_pdf(path, 2, 2)
        self.assertEqual(result.total_pages, 458)
        self.assertEqual(result.pages[0].number, 2)
        self.assertEqual(result.pages[0].status, "success")
        document = result.pages[0].document
        assert document is not None
        self.assertTrue(document.text.strip())


if __name__ == "__main__":
    _ = unittest.main()
