"""从已保存的 LlamaIndex 向量索引检索可核对的原文片段。

本模块只负责查询索引和整理结果，不重新解析 PDF。索引中的 Node 是正文和来源
元数据的持久化副本；查询结果因此即使源文件暂时不可用，也能显示保存的正文，
同时明确标记无法回到文件核验的结果。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import MetadataMode

from ..config.indexing import DEFAULT_EMBEDDING_DEVICE, HASH_READ_SIZE, MANIFEST_FILE, ManifestKey
from ..config.ingestion import MetadataKey
from ..indexing.model import snapshot_identity
from ..indexing.store import check_compatibility


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """一次检索命中的完整片段及其来源状态。

    ``score`` 是 LlamaIndex 向量存储返回的相似度分数，数值越大表示越相似；它
    只能在同一索引和同一查询配置下比较。``provenance_complete`` 为 ``False`` 时，
    仍可能有保存的正文，但不能据此宣称能够回到当前源文件核验。
    """

    rank: int
    score: float | None
    node_id: str
    text: str
    file_name: str
    source_path: str
    page_numbers: tuple[int, ...]
    source_exists: bool
    provenance_complete: bool
    provenance_warning: str | None = None

    @property
    def page_label(self) -> str:
        """返回适合命令行展示的单页或跨页标签。"""
        if len(self.page_numbers) == 1:
            return str(self.page_numbers[0])
        return f"{self.page_numbers[0]}–{self.page_numbers[-1]}"


def _read_manifest(index_path: Path) -> dict[str, Any]:
    """读取并检查项目自己的索引清单。"""
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


def _valid_provenance(metadata: dict[str, Any]) -> tuple[str, str, tuple[int, ...]] | None:
    """校验节点中的 PDF 来源字段，返回规范化值或 ``None``。"""
    file_name = metadata.get(MetadataKey.FILE_NAME)
    source_path = metadata.get(MetadataKey.SOURCE_PATH)
    pages = metadata.get(MetadataKey.PAGE_NUMBERS)
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    if not isinstance(pages, (list, tuple)) or not pages:
        return None
    if any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages):
        return None
    return file_name, source_path, tuple(pages)


def _source_status(source_path: str, manifest: dict[str, Any]) -> tuple[bool, bool, str | None]:
    """核对来源文件是否存在且仍是构建索引时记录的版本。"""
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

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(HASH_READ_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        return True, False, f"无法读取来源文件以核对版本: {source_path}: {exc}"
    if digest.hexdigest() != expected_digest:
        return True, False, f"来源文件内容已变化: {source_path}"
    return True, True, None


def _result_from_node(rank: int, node_with_score: Any, manifest: dict[str, Any]) -> RetrievalResult | None:
    """把 LlamaIndex 的命中对象转换为项目结果；来源不完整时丢弃。"""
    node = getattr(node_with_score, "node", None)
    if node is None:
        return None
    metadata = dict(getattr(node, "metadata", {}) or {})
    provenance = _valid_provenance(metadata)
    text = node.get_content(metadata_mode=MetadataMode.NONE)
    if provenance is None or not isinstance(text, str) or not text.strip():
        return None
    file_name, source_path, pages = provenance
    source_exists, provenance_complete, warning = _source_status(source_path, manifest)
    score = getattr(node_with_score, "score", None)
    if score is not None:
        score = float(score)
    return RetrievalResult(
        rank=rank,
        score=score,
        node_id=str(node.node_id),
        text=text,
        file_name=file_name,
        source_path=source_path,
        page_numbers=pages,
        source_exists=source_exists,
        provenance_complete=provenance_complete,
        provenance_warning=warning,
    )


def retrieve(index_path: Path, model_path: Path, question: str, top_k: int = 5) -> list[RetrievalResult]:
    """从持久化索引召回按相似度排序的可读片段。

    参数：
        index_path: ``indexing build`` 生成的索引目录。
        model_path: 与索引构建时完全相同的 Hugging Face 快照目录。
        question: 自然语言查询；空白问题会被拒绝。
        top_k: 最多请求的候选数量，必须为正整数。

    返回：
        按相似度从高到低排列的结果。来源字段格式非法的节点不会返回；源文件
        缺失的节点会保留正文并将 ``provenance_complete`` 设为 ``False``。

    异常：
        ``ValueError`` 表示索引、模型、查询配置或输入参数不可用。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Top-K 必须是正整数")

    index_path = index_path.expanduser().resolve()
    manifest = _read_manifest(index_path)
    model_path = model_path.expanduser().resolve(strict=True)
    identity = snapshot_identity(model_path)
    expected = {
        **identity,
        ManifestKey.DIMENSION: manifest.get(ManifestKey.DIMENSION),
        ManifestKey.CHUNK_SIZE: manifest.get(ManifestKey.CHUNK_SIZE),
        ManifestKey.CHUNK_OVERLAP: manifest.get(ManifestKey.CHUNK_OVERLAP),
    }
    check_compatibility(manifest, expected)

    # 与索引构建阶段使用同一设备和本地快照策略；查询文本不会发送到外部服务。
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    model = HuggingFaceEmbedding(model_name=str(model_path), device=DEFAULT_EMBEDDING_DEVICE, local_files_only=True)
    context = StorageContext.from_defaults(persist_dir=str(index_path))
    index = load_index_from_storage(context, embed_model=model)
    retriever = index.as_retriever(similarity_top_k=top_k, embed_model=model)
    raw_results = retriever.retrieve(question)

    results: list[RetrievalResult] = []
    for raw in raw_results:
        result = _result_from_node(len(results) + 1, raw, manifest)
        if result is not None:
            results.append(result)
    return results
