"""从已保存的 LlamaIndex 向量索引检索可核对的原文片段。

本模块只负责查询索引和整理结果，不重新解析 PDF。索引中的 Node 是正文和来源
元数据的持久化副本；查询结果因此即使源文件暂时不可用，也能显示保存的正文，
同时明确标记无法回到文件核验的结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import MetadataMode

from ..config.common import DEFAULT_INFERENCE_DEVICE
from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from ..config.ingestion import MetadataKey
from ..indexing.model import snapshot_identity
from ..indexing.store import check_compatibility
from ..provenance import format_pages, read_manifest, source_status


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
    original_rank: int | None = None
    rerank_score: float | None = None

    @property
    def page_label(self) -> str:
        """返回适合命令行展示的单页或跨页标签。"""
        return format_pages(self.page_numbers)

    @property
    def source_status_label(self) -> str:
        """返回来源是否可回到当前文件核对的展示文案。

        检索和回答命令都需要这行提示；放在结果对象上可以保证两处措辞一致，
        不会一处说“可核对”、另一处漏掉警告。
        """
        if self.provenance_complete:
            return "可回到当前 PDF 核对"
        return f"无法完成当前文件核对；{self.provenance_warning or '来源信息不完整'}"


def _valid_provenance(metadata: dict[str, Any]) -> tuple[str, str, tuple[int, ...]] | None:
    """校验节点中的 PDF 来源字段，返回规范化值或 ``None``。

    这里比切片写入侧 ``chunking.split`` 更宽松：索引中的 Node 可以跨页，且不强制
    要求 ``source_id``。两套规则分别守住“查询兼容性”和“写入完整性”，不要合并。
    """
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


def _result_from_node(
    rank: int,
    node_with_score: Any,
    manifest: dict[str, Any],
    source_status_cache: dict[str, tuple[bool, bool, str | None]],
) -> RetrievalResult | None:
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
    if source_path not in source_status_cache:
        source_status_cache[source_path] = source_status(source_path, manifest)
    source_exists, provenance_complete, warning = source_status_cache[source_path]
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


def retrieve(
    index_path: Path,
    model_path: Path,
    question: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievalResult]:
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
        ``ValueError`` 表示索引、模型身份、查询配置或输入参数不可用；
        模型目录不存在时由 ``Path.resolve(strict=True)`` 抛出 ``FileNotFoundError``。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("Top-K 必须是正整数")

    index_path = index_path.expanduser().resolve()
    manifest = read_manifest(index_path)
    model_path = model_path.expanduser().resolve(strict=True)
    identity = snapshot_identity(model_path)
    # 查询阶段能独立核对的只有模型身份（模型名、revision、快照内容摘要）。
    # 维度和切片参数只能从 manifest 自身读出，再拿它与 manifest 比较是恒真检查；
    # 模型身份一致时，这些派生配置必然来自同一次构建。
    check_compatibility(manifest, identity)

    # 与索引构建阶段使用同一设备和本地快照策略；查询文本不会发送到外部服务。
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    model = HuggingFaceEmbedding(model_name=str(model_path), device=DEFAULT_INFERENCE_DEVICE, local_files_only=True)
    context = StorageContext.from_defaults(persist_dir=str(index_path))
    index = load_index_from_storage(context, embed_model=model)
    retriever = index.as_retriever(similarity_top_k=top_k, embed_model=model)
    raw_results = retriever.retrieve(question)

    results: list[RetrievalResult] = []
    source_status_cache: dict[str, tuple[bool, bool, str | None]] = {}
    for raw in raw_results:
        result = _result_from_node(len(results) + 1, raw, manifest, source_status_cache)
        if result is not None:
            results.append(result)
    return results


def _validate_rerank_limit(value: int, label: str) -> None:
    """校验重排阶段使用的正整数限制。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} 必须是正整数")


def _load_cross_encoder(model_path: Path) -> Any:
    """加载本地 CrossEncoder，并明确禁止缺失模型时联网下载。"""
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"重排模型不存在或不是目录: {resolved}")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ValueError("缺少 sentence-transformers，无法加载本地重排模型") from exc
    return CrossEncoder(
        str(resolved),
        device=DEFAULT_INFERENCE_DEVICE,
        local_files_only=True,
        trust_remote_code=False,
    )


def rerank(
    question: str,
    candidates: Sequence[RetrievalResult],
    model_path: Path,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievalResult]:
    """用本地 CrossEncoder 对 dense 候选重新排序并返回前 ``top_k`` 条。

    dense 相似度保存在 ``score``，重排模型分数单独保存在 ``rerank_score``；
    ``original_rank`` 保留候选进入重排前的排名，避免把两个分数或两个排名混为一谈。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    _validate_rerank_limit(top_k, "Top-K")
    rerank_path = model_path.expanduser().resolve()
    if not rerank_path.is_dir():
        raise ValueError(f"重排模型不存在或不是目录: {rerank_path}")
    # 先判空再加载模型：dense 结果可能全部被来源校验过滤，此时没有可重排内容，
    # 不值得为返回空列表加载数 GB 的 CrossEncoder。模型路径错误仍在上面提前报出。
    if not candidates:
        return []
    model = _load_cross_encoder(rerank_path)

    pairs = [(question, candidate.text) for candidate in candidates]
    scores = model.predict(pairs)
    if len(scores) != len(candidates):
        raise ValueError("重排模型返回的分数数量与候选数量不一致")

    scored: list[tuple[int, RetrievalResult, float]] = []
    for index, (candidate, raw_score) in enumerate(zip(candidates, scores)):
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("重排模型必须为每个候选返回一个标量分数") from exc
        if not math.isfinite(score):
            raise ValueError("重排模型返回了非有限分数")
        scored.append((index, candidate, score))

    scored.sort(key=lambda item: (-item[2], item[0]))
    results: list[RetrievalResult] = []
    for rank, (_, candidate, score) in enumerate(scored[:top_k], start=1):
        original_rank = candidate.original_rank if candidate.original_rank is not None else candidate.rank
        results.append(replace(candidate, rank=rank, original_rank=original_rank, rerank_score=score))
    return results


def retrieve_and_rerank(
    index_path: Path,
    model_path: Path,
    rerank_model_path: Path,
    question: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievalResult]:
    """先进行 dense 粗召回，再用本地模型重排候选。

    ``candidate_k`` 控制进入重排的 dense 候选数量，``top_k`` 控制最终交付数量。
    候选不足时返回实际可用数量，不伪造结果；``top_k`` 不得大于 ``candidate_k``。
    """
    _validate_rerank_limit(candidate_k, "候选数量")
    _validate_rerank_limit(top_k, "Top-K")
    if top_k > candidate_k:
        raise ValueError("最终 Top-K 不能大于候选数量")
    rerank_path = rerank_model_path.expanduser().resolve()
    if not rerank_path.is_dir():
        raise ValueError(f"重排模型不存在或不是目录: {rerank_path}")
    candidates = retrieve(index_path, model_path, question, top_k=candidate_k)
    return rerank(question, candidates, rerank_path, top_k=top_k)
