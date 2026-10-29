"""标识本地 Hugging Face 模型快照及其实际文件内容。"""

import hashlib
from pathlib import Path

from ..config.indexing import (
    HASH_READ_SIZE, HF_CACHE_COMPONENT_SEPARATOR, HF_MODEL_CACHE_PREFIX,
    HF_SNAPSHOTS_DIR, SNAPSHOT_NAME_BYTEORDER, SNAPSHOT_NAME_LENGTH_BYTES, ManifestKey,
)


def snapshot_identity(path: Path) -> dict[str, str]:
    """读取缓存目录标识，并对文件路径和内容计算可重复的摘要。"""
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
