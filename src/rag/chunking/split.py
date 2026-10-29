"""将源文档切分为可溯源的 LlamaIndex 节点。"""

import logging
import time
from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from pathlib import Path

from ..config.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ..config.ingestion import MetadataKey
from .offset_splitter import split_with_offsets


logger = logging.getLogger(__name__)


def chunk_documents(documents: Sequence[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[BaseNode]:
    """将页面文档转换为带来源的节点；大小和重叠以 tokenizer 的 token 计。"""

    started = time.monotonic()
    logger.info("chunking.start documents=%s chunk_size=%s overlap=%s", len(documents), chunk_size, chunk_overlap)
    # 此限制比允许 chunk_overlap == chunk_size 的 SentenceSplitter 更严格。
    # 两者相等时合并窗口无法前进，重叠会退化为重复的文本块。
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切片大小必须大于 0，重叠量须大于等于 0 且小于切片大小")
    # 先按连续页面分组，再对每组切片，避免跨越缺页或不同文档。
    groups: list[list[Document]] = []
    for document in documents:
        metadata = document.metadata
        if "source_location" in metadata:
            raise ValueError("不再支持 source_location；PDF 来源请使用 page_numbers")
        required = (MetadataKey.FILE_NAME, MetadataKey.SOURCE_PATH, MetadataKey.SOURCE_ID)
        # 要求每个键对应非空字符串，以发现缺失的键、非字符串值和空白占位符；
        # 位置是否真实由写入方负责。
        if not all(isinstance(metadata.get(key.value), str) and metadata[key.value].strip()
                   for key in required):
            raise ValueError("来源身份或定位信息缺失，无法切片")
        if Path(metadata[MetadataKey.FILE_NAME.value]).suffix.lower() != ".pdf":
            raise ValueError("目前仅支持 PDF 来源切片")
        page_numbers = metadata.get(MetadataKey.PAGE_NUMBERS.value)
        # bool 是 int 的子类，页码必须排除 True 和 False。
        if not isinstance(page_numbers, list) or len(page_numbers) != 1 or type(page_numbers[0]) is not int or page_numbers[0] < 1:
            raise ValueError("PDF 来源页码缺失或无效")
        if not document.text.strip():
            continue
        page = (page_numbers[0] if isinstance(page_numbers, list) and len(page_numbers) == 1
                and type(page_numbers[0]) is int and page_numbers[0] > 0 else None)
        previous = groups[-1][-1] if groups else None
        previous_pages = previous.metadata.get(MetadataKey.PAGE_NUMBERS.value) if previous else None
        ignored = {MetadataKey.PAGE_NUMBERS.value}
        # 只有来源元数据和渲染设置一致的连续页面才能合并到同一切片；
        # 页码不连续或来源不同时保留各自的边界。
        same_source = previous is not None and (
            {k: v for k, v in previous.metadata.items() if k not in ignored}
            == {k: v for k, v in metadata.items() if k not in ignored}
        )
        if (same_source and previous is not None and page is not None
                and isinstance(previous_pages, list) and len(previous_pages) == 1
                and type(previous_pages[0]) is int and page == previous_pages[0] + 1
                and document.excluded_embed_metadata_keys == previous.excluded_embed_metadata_keys
                and document.excluded_llm_metadata_keys == previous.excluded_llm_metadata_keys):
            groups[-1].append(document)
        else:
            groups.append([document])

    nodes: list[BaseNode] = []
    for group in groups:
        document = group[0]
        metadata = document.metadata
        # 深拷贝后排除所有元数据键。SentenceSplitter 会考虑元数据：
        # 它从切片预算中扣除注入嵌入模型或 LLM 的元数据 token。排除后该
        # 字符串为空，因此 chunk_size 只衡量正文，结果也不再随本地路径长度变化。
        # 这只影响渲染后的文本，元数据本身仍保留在节点上。
        source = document.model_copy(deep=True)
        spans: list[tuple[int, int, int]] = []
        if len(group) > 1:
            # 在插入分隔符之前记录左闭右开的页面范围，
            # 以便将切片归属于实际相交的每一页。
            parts: list[str] = []
            offset = 0
            for page_document in group:
                if parts:
                    parts.append("\n")
                    offset += 1
                page_text = page_document.text
                spans.append((offset, offset + len(page_text), page_document.metadata[MetadataKey.PAGE_NUMBERS.value][0]))
                parts.append(page_text)
                offset += len(page_text)
            source.set_content("".join(parts))
            source.metadata.pop(MetadataKey.PAGE_NUMBERS.value, None)
        source.excluded_embed_metadata_keys = list(metadata)
        source.excluded_llm_metadata_keys = list(metadata)
        # 替换解析器推断的偏移量，后者可能指向更早出现的重复切片。
        page_nodes, offsets = split_with_offsets(source, chunk_size, chunk_overlap)
        # 上述临时排除设置会被复制到每个节点，从而使来源键无法进入嵌入文本和 LLM 上下文。
        # 恢复调用方的设置，使来源信息贯穿检索和回答。
        for node, (start_char, end_char) in zip(page_nodes, offsets):
            if not isinstance(node, TextNode):
                raise ValueError("无法确定切片来源位置")
            node.start_char_idx, node.end_char_idx = start_char, end_char
            if spans:
                # 左闭右开区间相交，表示该切片确实包含此页的字符。
                pages = [number for start, end, number in spans
                         if start < end_char and end > start_char]
                if not pages:
                    raise ValueError("无法确定切片来源页码")
                node.metadata[MetadataKey.PAGE_NUMBERS.value] = pages
            node.excluded_embed_metadata_keys = list(document.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = list(document.excluded_llm_metadata_keys)
        # 为每个节点设置实际页码后，刷新关联节点的元数据。
        for index, node in enumerate(page_nodes):
            if index:
                node.relationships[NodeRelationship.PREVIOUS] = page_nodes[index - 1].as_related_node_info()
            if index + 1 < len(page_nodes):
                node.relationships[NodeRelationship.NEXT] = page_nodes[index + 1].as_related_node_info()
        nodes.extend(page_nodes)
    logger.info("chunking.complete chunks=%s duration_seconds=%.3f", len(nodes), time.monotonic() - started)
    return nodes
