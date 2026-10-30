"""索引构建和本地嵌入模型共用的配置常量。

这些值集中放置，供索引 CLI、存储模块和模型身份校验共同使用，避免同一规则在多个
文件中各自维护。
"""

from enum import StrEnum
from typing import Literal

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
# 这是经验默认值；每个实际片段仍必须按所选模型的 token 上限逐片校验。
DEFAULT_INDEX_CHUNK_SIZE = 384

# 这些数值来自系统调用 ABI，不是用户配置，不能随模型或索引参数调整。
DARWIN_RENAME_EXCL = 0x4
LINUX_AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 1
MANIFEST_FILE = "manifest.json"
INDEX_BUILD_TEMP_PREFIX = ".index-building-"
INDEX_STAGING_DIR = "index"

# 下面的常量描述 Hugging Face 缓存目录约定，以及模型快照摘要的编码方式。
HF_SNAPSHOTS_DIR = "snapshots"
HF_MODEL_CACHE_PREFIX = "models--"
HF_CACHE_COMPONENT_SEPARATOR = "--"
SNAPSHOT_NAME_LENGTH_BYTES = 8
SNAPSHOT_NAME_BYTEORDER: Literal["big"] = "big"


class IndexBuildPhase(StrEnum):
    """索引构建阶段，用于在失败时说明错误发生在哪里。"""

    EMBEDDING = "向量化"
    PUBLISHING = "索引发布"


class IndexCommand(StrEnum):
    """索引 CLI 支持的三种操作：构建、概览和片段回查。"""

    BUILD = "build"
    OVERVIEW = "overview"
    READ = "read"


# 文件摘要按固定大小分块读取；当前索引阶段默认使用 CPU 执行本地推理。
HASH_READ_SIZE = 1024 * 1024
DEFAULT_EMBEDDING_DEVICE = "cpu"


class ManifestKey(StrEnum):
    """manifest、索引节点和文档来源记录共用的字段名。"""

    MODEL = "model"
    VERSION = "version"
    MODEL_SHA256 = "model_sha256"
    DIMENSION = "dimension"
    CHUNK_SIZE = "chunk_size"
    CHUNK_OVERLAP = "chunk_overlap"
    CHUNK_COUNT = "chunk_count"
    COMPLETE = "complete"
    DOCUMENTS = "documents"
    PATH = "path"
    SHA256 = "sha256"
    SOURCE_ID = "source_id"
    START = "start"
    END = "end"
    EMPTY_PAGES = "empty_pages"
    TEXT_COVERAGE_COMPLETE = "text_coverage_complete"


INDEX_COMPATIBILITY_KEYS = (
    ManifestKey.MODEL, ManifestKey.VERSION, ManifestKey.MODEL_SHA256,
    ManifestKey.DIMENSION, ManifestKey.CHUNK_SIZE, ManifestKey.CHUNK_OVERLAP,
)
