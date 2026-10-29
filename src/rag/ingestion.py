"""Load PDF pages as LlamaIndex documents with physical source locations."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader

from .ingestion_config import DEFAULT_START_PAGE, MetadataKey, PageStatus

@dataclass
class PageResult:
    number: int
    status: PageStatus
    document: Document | None = None
    error: str | None = None


@dataclass
class LoadResult:
    path: Path
    total_pages: int
    pages: list[PageResult]


class PagePDFReader(BaseReader):
    """Read pages independently so one extraction failure remains visible."""

    def load_pages(self, file: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
        """
        expanduser(): expand ~/docs/a.pdf to the real home directory.
        resolve(): make the path absolute and resolve .. and symlinks.
        strict=True: raise FileNotFoundError immediately if the path does not exist.
        """

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
                # extract_text() reads only the plain text layer; scanned PDFs (image pages) yield an empty string.
                text = reader.pages[number - 1].extract_text() or ""
                document = Document(
                    text=text,
                    metadata={
                        MetadataKey.FILE_NAME.value: path.name,
                        MetadataKey.SOURCE_PATH.value: str(path),
                        MetadataKey.SOURCE_ID.value: source_id,
                        MetadataKey.PAGE_NUMBER.value: number,
                    },
                )
                pages.append(PageResult(number, PageStatus.SUCCESS if text.strip() else PageStatus.EMPTY, document))
            except Exception as exc:
                pages.append(PageResult(number, PageStatus.FAILED, error=str(exc)))
        return LoadResult(path, total, pages)

    def load_data(self, file: Path, **kwargs) -> list[Document]:
        """
        Strategy reversal here: load_pages tolerates per-page failures so problems stay
        visible, while load_data (the standard entry point for the indexing flow) is
        all-or-nothing: any failed page raises and lists every failed page number.
        """
        result = self.load_pages(file, **kwargs)
        failures = [page for page in result.pages if page.status == PageStatus.FAILED]
        if failures:
            raise ValueError("页面解析失败: " + ", ".join(str(p.number) for p in failures))
        return [page.document for page in result.pages if page.document is not None]


# Module-level convenience function
def load_pdf(path: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
    return PagePDFReader().load_pages(path, start, end)
