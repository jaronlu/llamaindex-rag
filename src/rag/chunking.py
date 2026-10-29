"""Split source documents into traceable LlamaIndex nodes."""

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from .chunking_config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .ingestion_config import MetadataKey


def chunk_documents(documents: Sequence[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[BaseNode]:

    # Stricter than SentenceSplitter, which accepts chunk_overlap == chunk_size.
    # There the merge window cannot advance, so overlap degrades into duplicate blocks.
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("切片大小必须大于 0，重叠量须大于等于 0 且小于切片大小")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[BaseNode] = []
    for document in documents:
        metadata = document.metadata
        # PAGE_NUMBER is excluded on purpose: PDFs locate by physical page while other
        # formats use sections or offsets, so this layer must not bind to one format.
        required = (MetadataKey.FILE_NAME, MetadataKey.SOURCE_PATH,
                    MetadataKey.SOURCE_ID, MetadataKey.SOURCE_LOCATION)
        # Requiring a non-empty string per key catches missing keys, non-string values
        # and blank placeholders; whether a location is real is the writer's job.
        if not all(isinstance(metadata.get(key.value), str) and metadata[key.value].strip()
                   for key in required):
            raise ValueError("来源身份或定位信息缺失，无法切片")
        if not document.text.strip():
            continue

        # Deep-copy, then exclude every metadata key. SentenceSplitter is metadata-aware:
        # it subtracts metadata tokens injected into the embedding model or LLM from the
        # chunk budget. Excluding them empties that string, so chunk_size measures body
        # text only and results stop drifting with local path lengths. Only the rendered
        # text is affected; the metadata itself stays on the nodes.
        source = document.model_copy(deep=True)
        source.excluded_embed_metadata_keys = list(metadata)
        source.excluded_llm_metadata_keys = list(metadata)
        # Called per document: splits never overlap across documents, and PREVIOUS/NEXT
        # only link nodes from the same source.
        page_nodes = splitter.get_nodes_from_documents([source])
        # The temporary exclusions above are copied onto each node, which would keep source
        # keys out of embedding text and LLM context. Restore the caller's settings so that
        # sources carry through to retrieval and answering.
        for node in page_nodes:
            node.excluded_embed_metadata_keys = list(document.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = list(document.excluded_llm_metadata_keys)
        nodes.extend(page_nodes)
    return nodes
