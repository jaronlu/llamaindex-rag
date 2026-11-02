"""检索相关命令行参数的共享定义。

检索和回答命令重复声明同一组模型路径、候选数量和 Top-K 参数；集中注册后新增或
修改参数只需改一处。各命令的 ``--top-k`` 含义不同，因此帮助文本由调用方传入，
而不是在这里统一成一句模糊的说明。
"""

import argparse
from pathlib import Path

from ..config.common import DEFAULT_EMBEDDING_MODEL_PATH, DEFAULT_RERANK_MODEL_PATH
from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K


def add_retrieval_arguments(
    parser: argparse.ArgumentParser,
    *,
    top_k_help: str,
    dense_only_help: str | None = None,
) -> None:
    """向命令解析器注册检索和重排共用的参数。

    ``dense_only_help`` 为空时不注册 ``--dense-only``：评估命令始终要求重排候选
    路径，不能只跑 dense 后计算重排名次。
    """
    parser.add_argument("--model-path", type=Path, default=DEFAULT_EMBEDDING_MODEL_PATH, help="覆盖默认的 Hugging Face embedding 模型快照目录")
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help=top_k_help)
    parser.add_argument("--rerank-model-path", type=Path, default=DEFAULT_RERANK_MODEL_PATH, help="覆盖配置文件中的本地 CrossEncoder 模型目录")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K, help=f"重排前 dense 粗召回数量（默认 {DEFAULT_CANDIDATE_K}）")
    if dense_only_help is not None:
        parser.add_argument("--dense-only", action="store_true", help=dense_only_help)
