"""以人可读的方式检查 PDF 加载结果。"""

import argparse
import logging
import sys
from pathlib import Path

from .logging import logged_command

from ..ingestion.pdf import load_pdf
from ..config.ingestion import DEFAULT_START_PAGE, PageStatus


logger = logging.getLogger(__name__)


@logged_command
def main(argv: list[str] | None = None) -> int:
    """解析命令行参数，展示每页结果并返回进程状态码。"""
    parser = argparse.ArgumentParser(description="逐页加载 PDF 并显示原文及物理页码")
    parser.add_argument("file", type=Path, help="PDF 文件路径")
    parser.add_argument("--start", type=int, default=DEFAULT_START_PAGE, help="起始物理页码（从 1 开始）")
    parser.add_argument("--end", type=int, help="结束物理页码（包含该页；默认到末页）")
    args = parser.parse_args(argv)
    try:
        result = load_pdf(args.file, args.start, args.end)
    except Exception as exc:
        logger.exception("loading.failed file=%s", args.file)
        print(f"加载失败: {args.file}: {exc}", file=sys.stderr)
        return 1
    print(f"文件: {result.path}\nPDF 总页数: {result.total_pages}\n检查页数: {len(result.pages)}")
    for page in result.pages:
        print(f"\n--- 物理页码 {page.number} ---")
        if page.status == PageStatus.FAILED:
            print(f"解析失败: {page.error}")
        elif page.status == PageStatus.EMPTY:
            print("无可提取文本")
        elif page.document is not None:
            print(page.document.text)
        else:
            print("解析失败: 缺少页面文档", file=sys.stderr)
            return 1
    counts = {status: sum(page.status == status for page in result.pages)
              for status in PageStatus}
    print(f"\n统计: 成功 {counts[PageStatus.SUCCESS]} 页，空文本 {counts[PageStatus.EMPTY]} 页，失败 {counts[PageStatus.FAILED]} 页")
    return 1 if counts[PageStatus.FAILED] else 0


if __name__ == "__main__":
    sys.exit(main())
