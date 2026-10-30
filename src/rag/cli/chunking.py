"""提供命令行入口，展示切片正文、token 长度和来源页码。"""

import argparse
import logging
import sys
from pathlib import Path

from .logging import logged_command

from llama_index.core.schema import MetadataMode
from llama_index.core.utils import get_tokenizer

from ..chunking.split import chunk_documents
from ..config.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ..ingestion.pdf import load_pdf
from ..config.ingestion import DEFAULT_START_PAGE, MetadataKey, PageStatus


logger = logging.getLogger(__name__)


@logged_command
def main(argv: list[str] | None = None) -> int:
    """加载指定页面并展示切片、来源和长度统计。

    这个命令把 PDF 加载和切片模块串起来，输出完整片段供人工核对；返回值遵循
    命令行约定，0 表示处理完成，1 表示参数、加载或页面解析存在问题。
    """
    parser = argparse.ArgumentParser(description="切片并显示完整原文和来源")
    parser.add_argument("file", type=Path, help="PDF 文件路径")
    parser.add_argument("--start", type=int, default=DEFAULT_START_PAGE, help="起始物理页码")
    parser.add_argument("--end", type=int, help="结束物理页码（包含该页）")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="切片大小（token）")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="最大重叠量（token）")
    args = parser.parse_args(argv)
    try:
        result = load_pdf(args.file, args.start, args.end)
        documents = [p.document for p in result.pages if p.status == PageStatus.SUCCESS and p.document is not None]
        nodes = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    except Exception as exc:
        logger.exception("chunking.failed file=%s", args.file)
        print(f"切片失败: {args.file}: {exc}", file=sys.stderr)
        return 1
    print(f"文件: {result.path}\n切片大小: {args.chunk_size} token；最大重叠: {args.chunk_overlap} token")
    print("连续页面合并切片；按顺序展示相邻片段及其覆盖页码。")
    tokenize = get_tokenizer()
    texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
    lengths = [len(tokenize(text)) for text in texts]
    for index, (node, content, length) in enumerate(zip(nodes, texts, lengths), 1):
        pages = node.metadata[MetadataKey.PAGE_NUMBERS.value]
        location = str(pages[0]) if len(pages) == 1 else f"{pages[0]}–{pages[-1]}"
        print(f"\n--- 片段 {index} | ID {node.node_id} | 物理页码 {location} | {length} token | {len(content)} 字符 ---")
        print(content)
    for page in result.pages:
        if page.status != PageStatus.SUCCESS:
            print(f"物理页码 {page.number}: {'无可提取文本' if page.status == PageStatus.EMPTY else '解析失败: ' + str(page.error)}")
    print(f"\n统计: {len(nodes)} 个片段；token 长度最小/最大/平均: "
          + (f"{min(lengths)}/{max(lengths)}/{sum(lengths)/len(lengths):.1f}" if lengths else "无"))
    return 1 if any(page.status == PageStatus.FAILED for page in result.pages) else 0


if __name__ == "__main__":
    sys.exit(main())
