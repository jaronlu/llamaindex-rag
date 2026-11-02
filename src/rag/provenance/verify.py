"""跨阶段共享的来源标识、索引清单读取、来源核验和页码展示规则。

加载、切片、索引、检索和命令行原来各自复制了这些规则，副本之间的错误处理已经
出现分叉（例如 ``indexing read`` 在源文件不可读时会直接失败）。这里集中一份
实现，阶段模块只负责决定拿到结果后如何展示或失败。

本模块只依赖 ``config``；常量仍留在 ``config/indexing.py``，避免配置和业务规则
混在一起。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config.indexing import HASH_READ_SIZE, MANIFEST_FILE, ManifestKey


def file_sha256(path: Path) -> str:
    """按文件内容计算 SHA-256，分块读取以免一次把大文件载入内存。

    这里不捕获 ``OSError``：调用方最清楚“读不到文件”意味着什么——构建阶段应
    直接失败，检索阶段则可以降级为来源警告。返回值是十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_READ_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id_for(path: Path) -> str:
    """把规范化绝对路径转换成稳定的来源标识。

    标识基于路径而不是内容：文件内容是否变化由 manifest 中的 SHA-256 单独核对。
    因此文件移动后标识会变化，旧索引不会静默指向新位置，符合受控重建的约定。
    """
    return hashlib.sha256(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()


def read_manifest(index_path: Path) -> dict[str, Any]:
    """读取并检查项目自己的索引清单。

    只接受已完成构建的索引：目录、清单文件、JSON 对象和完成标记缺一不可。
    这样检索、回查和概览不会把只写了一半的目录当成可用索引。
    """
    index_path = index_path.expanduser().resolve()
    if not index_path.is_dir():
        raise ValueError(f"索引不存在: {index_path}")
    manifest_path = index_path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"索引缺少清单文件: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"索引清单无法读取: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get(ManifestKey.COMPLETE) is not True:
        raise ValueError("索引不完整，无法检索")
    return manifest


def source_status(source_path: str, manifest: dict[str, Any]) -> tuple[bool, bool, str | None]:
    """核对来源文件是否存在且仍是构建索引时记录的版本。

    返回 ``(文件存在, 可完整核对, 警告)``。文件存在但内容或记录不完整时仍返回
    已保存的正文，只是标记为不可核对，避免把“有页码”当成“已验证”。
    """
    path = Path(source_path)
    if not path.is_file():
        return False, False, f"来源文件不存在或已移动: {source_path}"

    records = manifest.get(ManifestKey.DOCUMENTS)
    record = next(
        (item for item in records if isinstance(item, dict) and item.get(ManifestKey.PATH) == source_path),
        None,
    ) if isinstance(records, list) else None
    expected_digest = record.get(ManifestKey.SHA256) if record else None
    if not isinstance(expected_digest, str) or not expected_digest:
        return True, False, f"索引清单缺少来源版本摘要: {source_path}"

    try:
        digest = file_sha256(path)
    except OSError as exc:
        return True, False, f"无法读取来源文件以核对版本: {source_path}: {exc}"
    if digest != expected_digest:
        return True, False, f"来源文件内容已变化: {source_path}"
    return True, True, None


def format_pages(pages: Sequence[int]) -> str:
    """把物理页码格式化成 ``291``、``291–292``；没有页码时返回“无”。"""
    if not pages:
        return "无"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}–{pages[-1]}"
