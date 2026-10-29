"""构建不可覆盖的本地向量索引，并按节点编号读取原文。"""

import ctypes
import errno
import json
import os
import sys
import math
import tempfile
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode, MetadataMode

from ..config.indexing import (
    DARWIN_RENAME_EXCL, LINUX_AT_FDCWD, LINUX_RENAME_NOREPLACE,
    INDEX_BUILD_TEMP_PREFIX, INDEX_COMPATIBILITY_KEYS, INDEX_STAGING_DIR,
    MANIFEST_FILE, IndexBuildPhase, ManifestKey,
)


class IndexBuildError(ValueError):
    """包含向量化统计和发布状态的构建错误。"""


def publish_index(staging: Path, destination: Path) -> None:
    """以禁止替换的系统调用原子发布目录。"""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(staging), os.fsencode(destination), DARWIN_RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise OSError("系统不支持安全的原子索引发布") from exc
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(LINUX_AT_FDCWD, os.fsencode(staging), LINUX_AT_FDCWD,
                        os.fsencode(destination), LINUX_RENAME_NOREPLACE)
    else:
        raise OSError("系统不支持安全的原子索引发布")
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise ValueError(f"索引位置已存在: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def build_index(nodes: Sequence[BaseNode], embed_model: BaseEmbedding, location: Path,
                config: dict[str, Any]) -> dict[str, Any]:
    """在临时目录完成构建，成功后发布；禁止覆盖已有索引。"""
    location = Path(location).expanduser().absolute()
    if location.exists():
        raise ValueError(f"索引位置已存在: {location}；请使用新目录进行受控重建")
    if not nodes:
        raise ValueError("没有可索引的片段")
    # 底层存储按节点 ID 索引，重复 ID 会覆盖先前片段并使统计失真。
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("存在重复的片段 ID")
    location.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    dimension = None
    prepared = []
    phase = IndexBuildPhase.EMBEDDING
    published = False
    try:
        for node in nodes:
            copy = node.model_copy(deep=True)
            # 仅嵌入正文，避免本地路径等来源信息影响向量或发送到模型。
            vector = embed_model.get_text_embedding(copy.get_content(metadata_mode=MetadataMode.NONE))
            if not vector or any(not math.isfinite(value) for value in vector):
                raise ValueError("模型返回空向量或非有限数值")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("模型返回了不同维度的向量")
            dimension = len(vector)
            copy.embedding = vector
            prepared.append(copy)
            completed += 1
        phase = IndexBuildPhase.PUBLISHING
        with tempfile.TemporaryDirectory(prefix=INDEX_BUILD_TEMP_PREFIX, dir=location.parent) as temp:
            staging = Path(temp) / INDEX_STAGING_DIR
            index = VectorStoreIndex(prepared, embed_model=embed_model)
            index.storage_context.persist(persist_dir=str(staging))
            manifest = {**config, ManifestKey.DIMENSION: dimension, ManifestKey.CHUNK_COUNT: len(nodes), ManifestKey.COMPLETE: True}
            (staging / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # 发布完成后临时目录清理仍可能失败，须保留真实的发布状态。
            publish_index(staging, location)
            published = True
        return manifest
    except Exception as exc:
        if published:
            raise IndexBuildError(f"索引已发布，但构建临时目录清理失败；向量化成功 {completed}。原因: {exc}") from exc
        # 向量化逐个执行；发布阶段失败时所有片段均已向量化。
        failed = 1 if phase == IndexBuildPhase.EMBEDDING else 0
        remaining = len(nodes) - completed - failed
        raise IndexBuildError(f"{phase}失败：向量化成功 {completed}，向量化失败 {failed}，未处理 {remaining}；索引未发布、不可用。原因: {exc}") from exc


def read_node(location: Path, node_id: str) -> BaseNode:
    """从持久化文档库查回指定片段，不读取源 PDF。"""
    if not location.exists():
        raise ValueError(f"索引不存在: {location}")
    manifest = json.loads((location / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get(ManifestKey.COMPLETE) is not True:
        raise ValueError("索引不完整，无法读取")
    context = StorageContext.from_defaults(persist_dir=str(location))
    node = context.docstore.get_node(node_id, raise_error=False)
    if node is None:
        raise ValueError(f"片段不存在: {node_id}")
    return node


def check_compatibility(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    """拒绝模型、维度或切片配置不一致的索引。"""
    mismatches = [key for key in INDEX_COMPATIBILITY_KEYS if key not in saved or key not in expected or saved[key] != expected[key]]
    if mismatches:
        raise ValueError("索引配置不兼容: " + ", ".join(mismatches) + "；请重建或使用独立索引")
