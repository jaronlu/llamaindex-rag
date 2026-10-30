"""跨阶段共用的本地模型路径和推理设备配置。"""

import os
from pathlib import Path


def _local_resource_path(variable: str, default: Path) -> Path:
    """读取可用环境变量覆盖的本地路径，并展开 ``~``。

    两个模型路径共用同一解析规则，避免日后修复空值或相对路径时只改到其中一条，
    导致 embedding 与 rerank 的配置行为悄悄分叉。
    """
    return Path(os.environ.get(variable, str(default))).expanduser()


DEFAULT_EMBEDDING_MODEL_PATH = _local_resource_path(
    "RAG_EMBEDDING_MODEL_PATH",
    Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620",
)

DEFAULT_RERANK_MODEL_PATH = _local_resource_path(
    "RAG_RERANK_MODEL_PATH",
    Path.home() / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
)

# 当前索引、查询和重排均使用本地 CPU 推理，统一设备默认值以保证可复现。
DEFAULT_INFERENCE_DEVICE = "cpu"
