"""在合并切分结果时记录偏移量的 SentenceSplitter。"""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.sentence import _Split
from llama_index.core.schema import BaseNode, Document
from pydantic import PrivateAttr


class _OffsetSentenceSplitter(SentenceSplitter):
    # 此适配器遵循已安装版本 SentenceSplitter 的私有合并约定。
    _offsets: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _split_text_value: str = PrivateAttr(default="")

    def _merge(self, splits: list[_Split], chunk_size: int) -> list[str]:
        self._offsets = []
        self._split_text_value = "".join(split.text for split in splits)
        chunks: list[str] = []
        # 保留每个切分结果的原始字符范围，包括复用的重叠部分。
        current: list[tuple[str, int, int, int]] = []
        size = 0
        position = 0
        new_chunk = True

        def close() -> None:
            nonlocal current, size, new_chunk
            raw = "".join(item[0] for item in current)
            trimmed = raw.strip()
            if trimmed:
                # 偏移量必须对应返回给 LlamaIndex 的去除首尾空白后的文本。
                start = current[0][2] + len(raw) - len(raw.lstrip())
                self._offsets.append((start, start + len(trimmed)))
                chunks.append(trimmed)
            old = current
            current = []
            size = 0
            # 复用完整的尾部切分结果，保持原有重叠行为。
            for item in reversed(old):
                if size + item[1] > self.chunk_overlap:
                    break
                current.insert(0, item)
                size += item[1]
            new_chunk = True

        for split in splits:
            if split.token_size > chunk_size:
                raise ValueError("Single token exceeded chunk size")
            if size + split.token_size > chunk_size and not new_chunk:
                close()
            while new_chunk and current and size + split.token_size > chunk_size:
                size -= current.pop(0)[1]
            current.append((split.text, split.token_size, position, position + len(split.text)))
            size += split.token_size
            position += len(split.text)
            new_chunk = False
        if not new_chunk:
            close()
        return chunks

    def _validated_offsets(self, text: str, chunks: list[str]) -> list[tuple[int, int]]:
        # 如果上游切分或后处理未保留原文，则拒绝处理；
        # 文段重复时，事后搜索切片无法确定位置。
        if self._split_text_value != text or len(chunks) != len(self._offsets) or any(text[start:end] != chunk for (start, end), chunk in zip(self._offsets, chunks)):
            raise ValueError("无法确定切片来源位置")
        return self._offsets


def split_with_offsets(document: Document, chunk_size: int, chunk_overlap: int) -> tuple[list[BaseNode], list[tuple[int, int]]]:
    """使用独立切分器一次性返回节点及其原文偏移量。"""
    splitter = _OffsetSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([document])
    offsets = splitter._validated_offsets(document.text, [node.get_content() for node in nodes])
    return nodes, offsets
