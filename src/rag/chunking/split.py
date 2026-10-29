"""Split source documents into traceable LlamaIndex nodes."""

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from pathlib import Path

from ..config.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ..config.ingestion import MetadataKey
from .offset_splitter import OffsetSentenceSplitter


def chunk_documents(documents: Sequence[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[BaseNode]:

    # Stricter than SentenceSplitter, which accepts chunk_overlap == chunk_size.
    # There the merge window cannot advance, so overlap degrades into duplicate blocks.
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切片大小必须大于 0，重叠量须大于等于 0 且小于切片大小")
    splitter = OffsetSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    groups: list[list[Document]] = []
    for document in documents:
        metadata = document.metadata
        # PAGE_NUMBERS is excluded on purpose: PDFs locate by physical page while other
        # formats use sections or offsets, so this layer must not bind to one format.
        required = (MetadataKey.FILE_NAME, MetadataKey.SOURCE_PATH, MetadataKey.SOURCE_ID)
        # Requiring a non-empty string per key catches missing keys, non-string values
        # and blank placeholders; whether a location is real is the writer's job.
        if not all(isinstance(metadata.get(key.value), str) and metadata[key.value].strip()
                   for key in required):
            raise ValueError("来源身份或定位信息缺失，无法切片")
        is_pdf = Path(metadata[MetadataKey.FILE_NAME.value]).suffix.lower() == ".pdf"
        page_numbers = metadata.get(MetadataKey.PAGE_NUMBERS.value)
        if is_pdf:
            if not isinstance(page_numbers, list) or len(page_numbers) != 1 or type(page_numbers[0]) is not int or page_numbers[0] < 1:
                raise ValueError("PDF 来源页码缺失或无效")
            # PDF page numbers are authoritative; do not retain a conflicting label.
            document = document.model_copy(deep=True)
            document.metadata.pop(MetadataKey.SOURCE_LOCATION.value, None)
            metadata = document.metadata
        elif not isinstance(metadata.get(MetadataKey.SOURCE_LOCATION.value), str) or not metadata[MetadataKey.SOURCE_LOCATION.value].strip():
            raise ValueError("来源定位信息缺失，无法切片")
        if not document.text.strip():
            continue
        page = (page_numbers[0] if isinstance(page_numbers, list) and len(page_numbers) == 1
                and type(page_numbers[0]) is int and page_numbers[0] > 0 else None)
        previous = groups[-1][-1] if groups else None
        previous_pages = previous.metadata.get(MetadataKey.PAGE_NUMBERS.value) if previous else None
        ignored = {MetadataKey.PAGE_NUMBERS.value, MetadataKey.SOURCE_LOCATION.value}
        # Only consecutive pages with matching source metadata and rendering settings
        # may share a chunk; gaps and differing sources keep separate boundaries.
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
        # Deep-copy, then exclude every metadata key. SentenceSplitter is metadata-aware:
        # it subtracts metadata tokens injected into the embedding model or LLM from the
        # chunk budget. Excluding them empties that string, so chunk_size measures body
        # text only and results stop drifting with local path lengths. Only the rendered
        # text is affected; the metadata itself stays on the nodes.
        source = document.model_copy(deep=True)
        spans: list[tuple[int, int, int]] = []
        if len(group) > 1:
            # Record half-open page ranges before inserting separators, so a chunk can
            # be attributed to every page it actually intersects.
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
            source.metadata.pop(MetadataKey.SOURCE_LOCATION.value, None)
        source.excluded_embed_metadata_keys = list(metadata)
        source.excluded_llm_metadata_keys = list(metadata)
        page_nodes = splitter.get_nodes_from_documents([source])
        # Replace parser-inferred offsets, which can point to an earlier repeated chunk.
        offsets = splitter.offsets(source.text, [node.get_content() for node in page_nodes])
        # The temporary exclusions above are copied onto each node, which would keep source
        # keys out of embedding text and LLM context. Restore the caller's settings so that
        # sources carry through to retrieval and answering.
        for node, (start_char, end_char) in zip(page_nodes, offsets):
            if not isinstance(node, TextNode):
                raise ValueError("无法确定切片来源位置")
            node.start_char_idx, node.end_char_idx = start_char, end_char
            if spans:
                pages = [number for start, end, number in spans
                         if start < end_char and end > start_char]
                if not pages:
                    raise ValueError("无法确定切片来源页码")
                node.metadata[MetadataKey.PAGE_NUMBERS.value] = pages
            node.excluded_embed_metadata_keys = list(document.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = list(document.excluded_llm_metadata_keys)
        # Refresh related-node metadata after assigning each node's actual page numbers.
        for index, node in enumerate(page_nodes):
            if index:
                node.relationships[NodeRelationship.PREVIOUS] = page_nodes[index - 1].as_related_node_info()
            if index + 1 < len(page_nodes):
                node.relationships[NodeRelationship.NEXT] = page_nodes[index + 1].as_related_node_info()
        nodes.extend(page_nodes)
    return nodes
