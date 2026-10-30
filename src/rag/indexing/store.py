"""本地向量索引的保存和读取。

这里没有重新实现向量数据库，真正的索引功能由 LlamaIndex 提供：

* ``VectorStoreIndex`` 负责组织向量和节点；
* ``StorageContext.persist`` 负责把它们写入磁盘；
* ``StorageContext.from_defaults`` 负责从磁盘重新打开它们。

本模块负责的是项目自己的保护措施，例如：不覆盖旧索引、先写临时目录再发布、
保存 ``manifest.json``，以及在失败时说明索引到底有没有发布成功。
"""

import ctypes
import errno
import json
import logging
import time
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


logger = logging.getLogger(__name__)


class IndexBuildError(ValueError):
    """索引构建失败时使用的错误。

    ``published`` 很重要：通常发生异常就意味着“没有索引”，但如果索引已经发布，
    只是后续清理临时目录时出错，就不能误报为“索引没有生成”。
    """

    def __init__(self, message: str, *, published: bool) -> None:
        super().__init__(message)
        self.published = published


def publish_index(staging: Path, destination: Path) -> None:
    """把临时索引目录一次性发布到最终位置，并且禁止覆盖旧目录。

    ``staging`` 是临时目录，里面应该已经有完整的 LlamaIndex 文件和 manifest。
    发布成功后，读取方看到的要么是旧的完整索引，要么是新的完整索引，
    不会看到只写了一半的目录。

    这里直接调用操作系统提供的原子重命名接口，而不是先判断目录是否存在、
    再普通重命名。后者在两个进程同时构建时会有竞态条件，可能互相覆盖。
    """
    # 这个系统调用会在同一个操作中完成“目标不存在”检查和重命名，避免并发覆盖。
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
    """构建并保存一个本地向量索引。

    参数说明：

    * ``nodes``：上游切片阶段产生的文本片段；
    * ``embed_model``：把每个片段的正文转换成数字向量的模型；
    * ``location``：最终索引目录；
    * ``config``：模型、切片参数和文档来源等项目元数据。

    整体流程是：先检查输入 → 逐个生成向量 → 在临时目录调用 LlamaIndex 持久化
    → 写入项目自己的 manifest → 原子发布到最终目录。任何一步失败，最终目录都
    不会出现半成品；已有目录也不会被覆盖。
    """
    started = time.monotonic()
    logger.info("index.start location=%s chunks=%s", location, len(nodes))
    location = Path(location).expanduser().absolute()
    if location.exists():
        raise ValueError(f"索引位置已存在: {location}；请使用新目录进行受控重建")
    if not nodes:
        raise ValueError("没有可索引的片段")
    # LlamaIndex 的文档库按 node_id 找节点；重复 ID 可能覆盖前一个片段，
    # 所以在交给 LlamaIndex 之前先拒绝重复 ID。
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
            # 在副本上写入 embedding，避免偷偷修改调用方手里的原始 Node。
            copy = node.model_copy(deep=True)
            # 只把正文交给嵌入模型。
            # source_path、页码等 metadata 用于溯源，不应该改变语义向量，
            # 也不应该因为生成向量而被发送给外部模型服务。
            vector = embed_model.get_text_embedding(copy.get_content(metadata_mode=MetadataMode.NONE))
            # 向量必须存在，而且每个数字都必须是有限值；否则后面的相似度计算
            # 可能得到无效结果或很难定位的错误。
            if not vector or any(not math.isfinite(value) for value in vector):
                raise ValueError("模型返回空向量或非有限数值")
            # 一个索引里的所有向量必须有相同长度，例如都为 384 维。
            if dimension is not None and len(vector) != dimension:
                raise ValueError("模型返回了不同维度的向量")
            dimension = len(vector)
            copy.embedding = vector
            prepared.append(copy)
            completed += 1
            logger.debug("index.embedding_progress completed=%s total=%s node_id=%s", completed, len(nodes), node.node_id)
        logger.info("index.embedding_complete completed=%s dimension=%s", completed, dimension)
        phase = IndexBuildPhase.PUBLISHING
        with tempfile.TemporaryDirectory(prefix=INDEX_BUILD_TEMP_PREFIX, dir=location.parent) as temp:
            staging = Path(temp) / INDEX_STAGING_DIR
            # 到这里每个 Node 都已经有 embedding。
            # 下面由 LlamaIndex 创建 VectorStoreIndex，并把向量和节点文档库保存下来。
            index = VectorStoreIndex(prepared, embed_model=embed_model)
            index.storage_context.persist(persist_dir=str(staging))
            # manifest 不是 LlamaIndex 的向量存储，而是项目自己的构建记录。
            # 它保存模型、维度、切片参数、文档来源和完成标记，方便之后检查索引
            # 是否可用，以及查询时确认配置和来源是否匹配。
            manifest = {**config, ManifestKey.DIMENSION: dimension, ManifestKey.CHUNK_COUNT: len(nodes), ManifestKey.COMPLETE: True}
            (staging / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # 只有 LlamaIndex 文件和 manifest 都写完，才把整个目录发布出去。
            publish_index(staging, location)
            published = True
            logger.info("index.published location=%s completed=%s", location, completed)
        logger.info("index.complete location=%s duration_seconds=%.3f", location, time.monotonic() - started)
        return manifest
    except Exception as exc:
        # 日志和错误信息要区分：当前片段失败、后面片段尚未处理，还是已经进入发布阶段。
        failed_count = 1 if phase == IndexBuildPhase.EMBEDDING and not published else 0
        logger.exception("index.failed location=%s phase=%s completed=%s failed=%s remaining=%s published=%s", location, phase, completed, failed_count, len(nodes) - completed - failed_count, published)
        if published:
            raise IndexBuildError(f"索引已发布，但构建临时目录清理失败；向量化成功 {completed}。原因: {exc}", published=True) from exc
        # 向量化是逐个进行的，所以可以报告“成功、失败、未处理”三种数量。
        # 如果已经进入发布阶段，则说明所有片段都已经完成向量化。
        failed = 1 if phase == IndexBuildPhase.EMBEDDING else 0
        remaining = len(nodes) - completed - failed
        raise IndexBuildError(f"{phase}失败：向量化成功 {completed}，向量化失败 {failed}，未处理 {remaining}；索引未发布、不可用。原因: {exc}", published=False) from exc


def read_node(location: Path, node_id: str) -> BaseNode:
    """从已保存的索引中按 ``node_id`` 取回一个片段。

    ``StorageContext`` 和 ``docstore.get_node`` 是 LlamaIndex 提供的读取能力。
    这里额外先检查 ``manifest.json`` 的完成标记，避免打开一个没有完整发布的目录。
    读取的是索引中保存的 Node，因此不会重新读取源 PDF；Node 里的正文和来源元数据
    可以在源文件不可用时仍然用于核对。
    """
    if not location.exists():
        raise ValueError(f"索引不存在: {location}")
    manifest = json.loads((location / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get(ManifestKey.COMPLETE) is not True:
        raise ValueError("索引不完整，无法读取")
    # LlamaIndex 从这个目录恢复 docstore、向量存储等持久化对象。
    context = StorageContext.from_defaults(persist_dir=str(location))
    # docstore 是 LlamaIndex 保存节点正文和 metadata 的地方；这里只按 ID 查一个节点。
    node = context.docstore.get_node(node_id, raise_error=False)
    if node is None:
        raise ValueError(f"片段不存在: {node_id}")
    return node


def check_compatibility(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    """检查已保存索引能否与当前查询配置一起使用。

    LlamaIndex 可以打开一个目录，但它不知道这个目录是不是由同一个模型和切片参数
    生成的。模型、模型文件版本、向量维度或切片参数不一致时，查询结果可能没有意义，
    因此由项目代码主动拒绝。函数没有返回值；没有异常就表示检查通过。
    """
    mismatches = [key for key in INDEX_COMPATIBILITY_KEYS if key not in saved or key not in expected or saved[key] != expected[key]]
    if mismatches:
        raise ValueError("索引配置不兼容: " + ", ".join(mismatches) + "；请重建或使用独立索引")
