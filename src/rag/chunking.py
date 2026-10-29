"""Split source documents into traceable LlamaIndex nodes."""

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from .chunking_config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .ingestion_config import MetadataKey


def chunk_documents(documents: Sequence[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[BaseNode]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切片大小必须大于 0，重叠量须大于等于 0 且小于切片大小")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[BaseNode] = []
    for document in documents:
        metadata = document.metadata
        required = (MetadataKey.FILE_NAME, MetadataKey.SOURCE_PATH,
                    MetadataKey.SOURCE_ID, MetadataKey.SOURCE_LOCATION)
        if not all(isinstance(metadata.get(key.value), str) and metadata[key.value].strip()
                   for key in required):
            raise ValueError("来源身份或定位信息缺失，无法切片")
        if not document.text.strip():
            continue
        # Size refers to text only; preserve the original metadata exclusions on nodes.
        source = document.model_copy(deep=True)
        source.excluded_embed_metadata_keys = list(metadata)
        source.excluded_llm_metadata_keys = list(metadata)
        page_nodes = splitter.get_nodes_from_documents([source])
        for node in page_nodes:
            node.excluded_embed_metadata_keys = list(document.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = list(document.excluded_llm_metadata_keys)
        nodes.extend(page_nodes)
    return nodes
