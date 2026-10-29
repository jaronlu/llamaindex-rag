"""索引与本地嵌入模型的常量。"""

from enum import StrEnum
from typing import Literal

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_REVISION = "7999e1d3359715c523056ef9478215996d62a620"

# 系统调用 ABI 定义的值，不随用户配置变化。
DARWIN_RENAME_EXCL = 0x4
LINUX_AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 1
MANIFEST_FILE = "manifest.json"
INDEX_BUILD_TEMP_PREFIX = ".index-building-"
INDEX_STAGING_DIR = "index"

# Hugging Face 缓存目录约定及模型快照摘要的编码格式。
HF_SNAPSHOTS_DIR = "snapshots"
HF_MODEL_CACHE_PREFIX = "models--"
HF_CACHE_COMPONENT_SEPARATOR = "--"
SNAPSHOT_NAME_LENGTH_BYTES = 8
SNAPSHOT_NAME_BYTEORDER: Literal["big"] = "big"


class IndexBuildPhase(StrEnum):
    """构建失败时用于区分统计的阶段。"""

    EMBEDDING = "向量化"
    PUBLISHING = "索引发布"


class IndexCommand(StrEnum):
    """索引命令行支持的操作。"""

    BUILD = "build"
    OVERVIEW = "overview"
    READ = "read"


# 文件摘要计算的读取块大小，以及本阶段默认的本地推理设备。
HASH_READ_SIZE = 1024 * 1024
DEFAULT_EMBEDDING_DEVICE = "cpu"


class ManifestKey(StrEnum):
    """持久化索引和文档记录中共享的字段名。"""

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
