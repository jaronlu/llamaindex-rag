"""检索命令行入口，展示候选片段、相似度和可核对来源。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .logging import logged_command

from ..retrieval.search import retrieve


logger = logging.getLogger(__name__)


@logged_command
def main(argv: list[str] | None = None) -> int:
    """从已保存索引检索问题并输出完整候选片段。

    命令不重新读取 PDF；正文和页码来自持久化 Node。源文件消失时仍展示索引中保存
    的正文，但会明确标记为无法回到当前文件核对，避免把缺失来源静默当作已验证。
    """
    parser = argparse.ArgumentParser(description="从持久化向量索引检索并显示可追溯片段")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="检索一个自然语言问题")
    search.add_argument("index", type=Path, help="索引目录")
    search.add_argument("question", help="自然语言问题")
    search.add_argument("--model-path", required=True, type=Path, help="与索引构建时相同的 Hugging Face 快照目录")
    search.add_argument("--top-k", type=int, default=5, help="最多显示的候选数量（默认 5）")
    args = parser.parse_args(argv)

    try:
        results = retrieve(args.index, args.model_path, args.question, args.top_k)
    except Exception as exc:
        logger.exception("retrieval.failed index=%s question_length=%s", args.index, len(args.question))
        print(f"检索失败: {exc}", file=sys.stderr)
        return 1

    print(f"问题: {args.question}")
    print(f"索引: {args.index.expanduser().resolve()}")
    print(f"Top-K: {args.top_k}")
    print("分数说明: 相似度分数越高表示向量越相似；分数只在同一索引和查询配置下比较。")
    if not results:
        print("未找到可交付的检索结果：没有候选片段通过来源和正文检查。")
        return 0

    for result in results:
        score = "无" if result.score is None else f"{result.score:.6f}"
        print(f"\n--- 结果 {result.rank} ---")
        print(f"排名: {result.rank}")
        print(f"相似度分数: {score}")
        print(f"片段 ID: {result.node_id}")
        print(f"文件名: {result.file_name}")
        print(f"来源路径: {result.source_path}")
        print(f"物理页码: {result.page_label}")
        if result.provenance_complete:
            print("来源状态: 可回到当前 PDF 核对")
        else:
            print(f"来源状态: 无法完成当前文件核对；{result.provenance_warning or '来源信息不完整'}")
        print("完整原文:")
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
