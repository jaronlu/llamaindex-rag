"""SentenceSplitter with offsets recorded while its splits are merged."""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.sentence import _Split
from pydantic import PrivateAttr


class OffsetSentenceSplitter(SentenceSplitter):
    # This adapter follows the installed SentenceSplitter's private merge contract.
    _offsets: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _split_text_value: str = PrivateAttr(default="")

    def _merge(self, splits: list[_Split], chunk_size: int) -> list[str]:
        self._offsets = []
        self._split_text_value = "".join(split.text for split in splits)
        chunks: list[str] = []
        # Keep each split's original character range, including reused overlap.
        current: list[tuple[str, int, int, int]] = []
        size = 0
        position = 0
        new_chunk = True

        def close() -> None:
            nonlocal current, size, new_chunk
            raw = "".join(item[0] for item in current)
            trimmed = raw.strip()
            if trimmed:
                # Offsets must describe the stripped text returned to LlamaIndex.
                start = current[0][2] + len(raw) - len(raw.lstrip())
                self._offsets.append((start, start + len(trimmed)))
                chunks.append(trimmed)
            old = current
            current = []
            size = 0
            # Reuse whole trailing splits, preserving the original overlap behavior.
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

    def offsets(self, text: str, chunks: list[str]) -> list[tuple[int, int]]:
        # Fail closed if upstream splitting or postprocessing stops preserving text;
        # searching for chunks afterward is ambiguous when passages repeat.
        if self._split_text_value != text or len(chunks) != len(self._offsets) or any(text[start:end] != chunk for (start, end), chunk in zip(self._offsets, chunks)):
            raise ValueError("无法确定切片来源位置")
        return self._offsets
