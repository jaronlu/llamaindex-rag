"""python -m src.rag 的默认入口：以索引为参数启动连续中文问答。

各阶段仍有独立命令；这里只把最常用的连续对话入口放到包级别，避免每次输入
完整的模块路径。参数和帮助与 cli.chat 完全一致。
"""

import sys

from .cli.chat import main


if __name__ == "__main__":
    sys.exit(main())
