"""逐页读取 PDF，并把页面转换成带物理来源位置的 LlamaIndex 文档。

``pypdf`` 负责从 PDF 中提取文本，LlamaIndex 的 ``Document`` 负责承载页面正文和
metadata；本模块额外保留每页的成功、空文本或失败状态，让上游可以发现解析缺口。
"""

import logging
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader

from ..config.ingestion import DEFAULT_START_PAGE, MetadataKey, PageStatus

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """记录一页 PDF 的处理结果。

    ``document`` 只在成功或空文本时保存该页的 LlamaIndex Document；``error`` 只在
    页面解析失败时保存异常原因。逐页记录状态可以避免单页失败被整体加载流程掩盖。
    """

    number: int
    status: PageStatus
    document: Document | None = None
    error: str | None = None


@dataclass
class LoadResult:
    """保存一次 PDF 加载的整体结果。

    ``pages`` 只包含请求范围内的页面，``total_pages`` 仍表示源 PDF 的总页数，
    因此调用方可以区分“未选择某页”和“某页为空或解析失败”。
    """

    path: Path
    total_pages: int
    pages: list[PageResult]


class PagePDFReader(BaseReader):
    """逐页读取 PDF 的 Reader。

    它实现 LlamaIndex ``BaseReader`` 的接口，同时提供项目需要的 ``load_pages``：
    后者不会因为一页失败就丢失其他页的结果，而是把失败状态显式返回。
    """

    def load_pages(self, file: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
        """读取物理页码闭区间 ``[start, end]``，并逐页返回结果。

        页码从 1 开始；单页提取失败会记录在 ``PageResult`` 中，其他页面仍继续处理。
        ``end`` 省略时读取到 PDF 末页。
        """
        started = time.monotonic()
        logger.info("loading.start file=%s start=%s end=%s", file, start, end)
        # 展开 ~ 并解析为真实绝对路径；strict=True 让不存在的路径立即报错。
        path = Path(file).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"不是有效的 PDF 文件: {path}")
        reader = PdfReader(path)
        total = len(reader.pages)
        end = total if end is None else end
        if start < 1 or end < start or end > total:
            raise ValueError(f"页码范围无效: {start}-{end}，文档共 {total} 页")
        # source_id 基于规范化路径而不是文件内容；文件移动后它会随路径变化。
        source_id = sha256(str(path).encode("utf-8")).hexdigest()
        pages = []
        for number in range(start, end + 1):
            try:
                # pypdf 的 extract_text() 只读取 PDF 的文本层；扫描版 PDF 的图片页通常为空。
                text = reader.pages[number - 1].extract_text() or ""
                # LlamaIndex Document 同时承载页面正文和来源 metadata，供后续切片使用。
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
                logger.exception("loading.page_failed file=%s page=%s", path, number)
                pages.append(PageResult(number, PageStatus.FAILED, error=str(exc)))
        logger.info("loading.complete file=%s total_pages=%s selected=%s success=%s empty=%s failed=%s duration_seconds=%.3f", path, total, len(pages), sum(p.status == PageStatus.SUCCESS for p in pages), sum(p.status == PageStatus.EMPTY for p in pages), sum(p.status == PageStatus.FAILED for p in pages), time.monotonic() - started)
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


# 模块级便捷函数：给命令行和其他模块提供不必实例化 Reader 的入口。
def load_pdf(path: Path, start: int = DEFAULT_START_PAGE, end: int | None = None) -> LoadResult:
    """加载指定 PDF 页面，并返回逐页结果供命令行及其他模块使用。"""
    return PagePDFReader().load_pages(path, start, end)
