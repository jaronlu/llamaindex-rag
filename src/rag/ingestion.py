"""Load PDF pages as LlamaIndex documents with physical source locations."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader


@dataclass
class PageResult:
    number: int
    status: str
    document: Document | None = None
    error: str | None = None


@dataclass
class LoadResult:
    path: Path
    total_pages: int
    pages: list[PageResult]


class PagePDFReader(BaseReader):
    """Read pages independently so one extraction failure remains visible."""

    def load_pages(self, file: Path, start: int = 1, end: int | None = None) -> LoadResult:
        path = Path(file).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"不是有效的 PDF 文件: {path}")
        reader = PdfReader(path)
        total = len(reader.pages)
        end = total if end is None else end
        if start < 1 or end < start or end > total:
            raise ValueError(f"页码范围无效: {start}-{end}，文档共 {total} 页")
        source_id = sha256(str(path).encode("utf-8")).hexdigest()
        pages = []
        for number in range(start, end + 1):
            try:
                text = reader.pages[number - 1].extract_text() or ""
                document = Document(
                    text=text,
                    metadata={"file_name": path.name, "source_path": str(path),
                              "source_id": source_id, "page_number": number},
                )
                pages.append(PageResult(number, "success" if text.strip() else "empty", document))
            except Exception as exc:
                pages.append(PageResult(number, "failed", error=str(exc)))
        return LoadResult(path, total, pages)

    def load_data(self, file: Path, **kwargs) -> list[Document]:
        result = self.load_pages(file, **kwargs)
        failures = [page for page in result.pages if page.status == "failed"]
        if failures:
            raise ValueError("页面解析失败: " + ", ".join(str(p.number) for p in failures))
        return [page.document for page in result.pages if page.document is not None]


def load_pdf(path: Path, start: int = 1, end: int | None = None) -> LoadResult:
    return PagePDFReader().load_pages(path, start, end)
