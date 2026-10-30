"""识别本地 Hugging Face 模型快照，并为其内容生成稳定摘要。

模型目录名和 revision 可以说明模型来源，但不能证明目录中的权重没有变化；
本模块因此还会对快照内的相对文件路径和文件内容计算摘要，供索引兼容性检查使用。
"""

import hashlib
from pathlib import Path

from ..config.indexing import (
    HASH_READ_SIZE, HF_CACHE_COMPONENT_SEPARATOR, HF_MODEL_CACHE_PREFIX,
    HF_SNAPSHOTS_DIR, SNAPSHOT_NAME_BYTEORDER, SNAPSHOT_NAME_LENGTH_BYTES, ManifestKey,
)


def snapshot_identity(path: Path) -> dict[str, str]:
    """读取 Hugging Face 缓存快照的身份信息和内容摘要。

    返回值包含模型名、快照版本和模型文件摘要。函数只接受标准缓存中的快照目录，
    并对其中所有文件排序后计算摘要，使同一组文件得到可重复的结果。
    """
    path = path.expanduser().resolve(strict=True)
    repo = path.parent.parent.name
    if not path.is_dir() or path.parent.name != HF_SNAPSHOTS_DIR or not repo.startswith(HF_MODEL_CACHE_PREFIX):
        raise ValueError("模型路径必须是 Hugging Face 缓存中的快照目录")
    parts = repo.removeprefix(HF_MODEL_CACHE_PREFIX).split(HF_CACHE_COMPONENT_SEPARATOR)
    if len(parts) != 2 or not all(parts) or not path.name:
        raise ValueError("无法识别模型快照身份")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("模型快照为空")
    digest = hashlib.sha256()
    for item in files:
        name = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(SNAPSHOT_NAME_LENGTH_BYTES, SNAPSHOT_NAME_BYTEORDER))
        digest.update(name)
        file_digest = hashlib.sha256()
        with item.open("rb") as source:
            for block in iter(lambda: source.read(HASH_READ_SIZE), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return {ManifestKey.MODEL: "/".join(parts), ManifestKey.VERSION: path.name, ManifestKey.MODEL_SHA256: digest.hexdigest()}
