"""把加载后的页面文档切分成带来源信息的 LlamaIndex 节点。

LlamaIndex 负责具体的句子切分和节点对象；本模块负责决定哪些页面可以合并、
校验来源元数据、恢复页码信息，并把相邻节点关系保存下来。
"""

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
    """把页面文档转换成带来源的节点。

    ``documents`` 通常由 PDF 加载阶段按页生成；``chunk_size`` 和 ``chunk_overlap``
    使用 LlamaIndex tokenizer 的 token 数。函数会保留源文件、物理页码和字符范围，
    使检索结果可以回到原文位置。
    """

    started = time.monotonic()
    logger.info("chunking.start documents=%s chunk_size=%s overlap=%s", len(documents), chunk_size, chunk_overlap)
    # 这里比 LlamaIndex 的 SentenceSplitter 更早拒绝 chunk_overlap == chunk_size。
    # 两者相等时窗口无法向前推进，可能反复产生相同的文本块。
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切片大小必须大于 0，重叠量须大于等于 0 且小于切片大小")
    # 先按来源相同且页码连续的页面分组，再对每组切片，避免跨越缺页或不同文档。
    groups: list[list[Document]] = []
    for document in documents:
        metadata = document.metadata
        if "source_location" in metadata:
            raise ValueError("不再支持 source_location；PDF 来源请使用 page_numbers")
        required = (MetadataKey.FILE_NAME, MetadataKey.SOURCE_PATH, MetadataKey.SOURCE_ID)
        # 每个来源字段都必须是非空字符串。这样可以尽早发现字段缺失、类型错误或
        # 空白占位符；路径和 source_id 是否真实有效则由写入方负责。
        if not all(isinstance(metadata.get(key.value), str) and metadata[key.value].strip()
                   for key in required):
            raise ValueError("来源身份或定位信息缺失，无法切片")
        if Path(metadata[MetadataKey.FILE_NAME.value]).suffix.lower() != ".pdf":
            raise ValueError("目前仅支持 PDF 来源切片")
        page_numbers = metadata.get(MetadataKey.PAGE_NUMBERS.value)
        # Python 中 bool 是 int 的子类，因此必须用 type(...) is int 排除 True 和 False。
        if not isinstance(page_numbers, list) or len(page_numbers) != 1 or type(page_numbers[0]) is not int or page_numbers[0] < 1:
            raise ValueError("PDF 来源页码缺失或无效")
        if not document.text.strip():
            continue
        # 上一段已保证 page_numbers 是单元素、正整数的列表，这里直接取页码。
        page = page_numbers[0]
        previous = groups[-1][-1] if groups else None
        previous_pages = previous.metadata.get(MetadataKey.PAGE_NUMBERS.value) if previous else None
        ignored = {MetadataKey.PAGE_NUMBERS.value}
        # 只有来源元数据、渲染设置都一致且页码连续的页面才能合并到同一组；
        # 页码不连续或来源不同时保留各自的边界，避免错误地跨文档拼接文本。
        same_source = previous is not None and (
            {k: v for k, v in previous.metadata.items() if k not in ignored}
            == {k: v for k, v in metadata.items() if k not in ignored}
        )
        if (same_source and previous is not None
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
        # 在副本上暂时排除所有 metadata。LlamaIndex 的 SentenceSplitter 会把要注入
        # 嵌入模型或 LLM 的 metadata token 算入预算；排除后 chunk_size 只衡量正文，
        # 不会因为本地路径长度不同而产生不同切片。原始 metadata 仍会保留在节点上。
        source = document.model_copy(deep=True)
        spans: list[tuple[int, int, int]] = []
        if len(group) > 1:
            # 在插入页面分隔符之前记录每页的左闭右开字符范围，
            # 之后可以把跨页切片归属于它实际相交的所有物理页。
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
        # 使用我们根据合并文本计算的偏移量，替换解析器可能因重复文本而推断出的旧位置。
        page_nodes, offsets = split_with_offsets(source, chunk_size, chunk_overlap)
        # 上面的临时排除设置会被复制到每个节点；这里恢复调用方原本的设置，
        # 让来源 metadata 继续贯穿检索和回答流程。
        for node, (start_char, end_char) in zip(page_nodes, offsets):
            if not isinstance(node, TextNode):
                raise ValueError("无法确定切片来源位置")
            node.start_char_idx, node.end_char_idx = start_char, end_char
            if spans:
                # 两个左闭右开区间相交，说明这个切片确实包含该页的字符。
                pages = [number for start, end, number in spans
                         if start < end_char and end > start_char]
                if not pages:
                    raise ValueError("无法确定切片来源页码")
                node.metadata[MetadataKey.PAGE_NUMBERS.value] = pages
            node.excluded_embed_metadata_keys = list(document.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = list(document.excluded_llm_metadata_keys)
        # 页码已经确定后，补充前后节点关系，方便后续查看相邻上下文。
        for index, node in enumerate(page_nodes):
            if index:
                node.relationships[NodeRelationship.PREVIOUS] = page_nodes[index - 1].as_related_node_info()
            if index + 1 < len(page_nodes):
                node.relationships[NodeRelationship.NEXT] = page_nodes[index + 1].as_related_node_info()
        nodes.extend(page_nodes)
    logger.info("chunking.complete chunks=%s duration_seconds=%.3f", len(nodes), time.monotonic() - started)
    return nodes
