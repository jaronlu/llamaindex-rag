"""将 PDF 页面加载为带有物理来源位置的 LlamaIndex 文档。"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader

from ..config.ingestion import DEFAULT_START_PAGE, MetadataKey, PageStatus

@dataclass
class PageResult:
    """单页的物理页码、处理状态，以及成功时的文档或失败原因。"""

    number: int
    status: PageStatus
    document: Document | None = None
    error: str | None = None


@dataclass
class LoadResult:
    """一次加载的来源路径、PDF 总页数和所选页面的结果。"""

    path: Path
    total_pages: int
    pages: list[PageResult]


class PagePDFReader(BaseReader):
    """独立读取每一页，使单页提取失败保持可见。"""

    def load_pages(self, file: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
        """读取闭区间 [start, end]；逐页返回结果，不因单页失败中断。"""
        # 展开 ~ 并解析为真实绝对路径；strict=True 使不存在的路径立即报错。
        path = Path(file).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"不是有效的 PDF 文件: {path}")
        reader = PdfReader(path)
        total = len(reader.pages)
        end = total if end is None else end
        if start < 1 or end < start or end > total:
            raise ValueError(f"页码范围无效: {start}-{end}，文档共 {total} 页")
        # 身份基于路径而非文件内容；移动文件后身份会变化。
        source_id = sha256(str(path).encode("utf-8")).hexdigest()
        pages = []
        for number in range(start, end + 1):
            try:
                # extract_text() 仅读取纯文本层；扫描版 PDF（图片页面）会得到空字符串。
                text = reader.pages[number - 1].extract_text() or ""
                # Document 是页面文本和来源元数据的容器，供切片阶段使用。
                document = Document(
                    text=text,
                    metadata={
                        MetadataKey.FILE_NAME.value: path.name,
                        MetadataKey.SOURCE_PATH.value: str(path),
                        MetadataKey.SOURCE_ID.value: source_id,
                        MetadataKey.PAGE_NUMBERS.value: [number],
                    },
                )
                pages.append(PageResult(number, PageStatus.SUCCESS if text.strip() else PageStatus.EMPTY, document))
            except Exception as exc:
                pages.append(PageResult(number, PageStatus.FAILED, error=str(exc)))
        return LoadResult(path, total, pages)

    def load_data(self, file: Path, **kwargs) -> list[Document]:
        """
        这是 BaseReader 的接口；只要有页面失败就抛出异常。
        当前命令行使用 load_pages，以便检查每页的状态。
        """
        result = self.load_pages(file, **kwargs)
        failures = [page for page in result.pages if page.status == PageStatus.FAILED]
        if failures:
            raise ValueError("页面解析失败: " + ", ".join(str(p.number) for p in failures))
        return [page.document for page in result.pages if page.document is not None]


# 模块级便捷函数
def load_pdf(path: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
    """返回所选页面的加载结果，供命令行及其他模块使用。"""
    return PagePDFReader().load_pages(path, start, end)
