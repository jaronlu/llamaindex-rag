"""在 LlamaIndex 句子切分的基础上补充结构边界和原文字符位置。

LlamaIndex 的 ``SentenceSplitter`` 负责块内的句子切分和最终合并；本模块额外做两件事：
先按条目/项目符号结构划块，让片段边界优先落在条目之间；再记录每个片段在输入文本中
的左闭右开字符区间，供后续页码溯源使用。
"""

import re

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.sentence import _Split
from llama_index.core.schema import BaseNode, Document
from pydantic import PrivateAttr

from ..config.chunking import STRUCTURE_BLOCK_START_REGEX


def structure_block_starts(text: str) -> list[int]:
    """返回结构块起点在 ``text`` 中的字符位置，用于让切片优先在条目之间断开。

    起点来自行首的项目符号或编号条目；第一个块总是从 0 开始，末尾追加文本长度，
    便于直接和下一项组成左闭右开的区间。函数不改变原文，所以字符位置仍可用于溯源。
    """
    starts = [match.start() for match in re.finditer(STRUCTURE_BLOCK_START_REGEX, text, flags=re.MULTILINE)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(text))
    return starts


class _OffsetSentenceSplitter(SentenceSplitter):
    """沿用 LlamaIndex 的大小与重叠规则，加入结构边界并记录每个片段的字符范围。

    这是一个内部适配器：结构块由本类划分，块内句子切分与合并仍由 LlamaIndex 完成。
    它同时把即将返回的文本片段与原始拼接文本的位置对应起来。
    """

    # _merge 是已安装版本 SentenceSplitter 的内部方法，因此必须遵循该版本的
    # _Split 结构和合并约定；升级 LlamaIndex 后需要重新验证这里的适配代码。
    _offsets: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _split_text_value: str = PrivateAttr(default="")

    def _split(self, text: str, chunk_size: int) -> list[_Split]:
        """先按结构块切分，块内再沿用 LlamaIndex 的句子切分。

        结构块（条目、项目符号）不超过 ``chunk_size`` 时保持完整，使合并阶段只会
        在块之间断开；超长块交回父类按句子进一步切分，避免单个块撑爆切片上限。
        所有块都是原文的连续片段，因此后续偏移量校验仍然成立。
        """
        splits: list[_Split] = []
        starts = structure_block_starts(text)
        for begin, end in zip(starts, starts[1:]):
            if begin >= end:
                continue
            block = text[begin:end]
            token_size = self._token_size(block)
            if token_size <= chunk_size:
                splits.append(_Split(text=block, is_sentence=True, token_size=token_size))
            else:
                splits.extend(super()._split(block, chunk_size))
        return splits

    def _merge(self, splits: list[_Split], chunk_size: int) -> list[str]:
        self._offsets = []
        self._split_text_value = "".join(split.text for split in splits)
        chunks: list[str] = []
        # 保存每个原始切分结果的字符范围；重叠部分会在下一个窗口中再次使用。
        current: list[tuple[str, int, int, int]] = []
        size = 0
        position = 0
        new_chunk = True

        def close() -> None:
            # 输出当前窗口，并把不超过 chunk_overlap 的尾部切分结果留给下一个窗口。
            nonlocal current, size, new_chunk
            raw = "".join(item[0] for item in current)
            trimmed = raw.strip()
            if trimmed:
                # LlamaIndex 返回的是去掉首尾空白后的文本，所以偏移量也必须跳过同样的空白。
                start = current[0][2] + len(raw) - len(raw.lstrip())
                self._offsets.append((start, start + len(trimmed)))
                chunks.append(trimmed)
            old = current
            current = []
            size = 0
            # 复用完整的尾部切分结果，保持 SentenceSplitter 原本的重叠行为。
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
        # 如果上游切分或后处理改变了原文，就不能安全地把片段定位回去。
        # 不能用 text.find(chunk) 事后搜索，因为重复文段会产生多个可能位置。
        if self._split_text_value != text or len(chunks) != len(self._offsets) or any(text[start:end] != chunk for (start, end), chunk in zip(self._offsets, chunks)):
            raise ValueError("无法确定切片来源位置")
        return self._offsets


def split_with_offsets(document: Document, chunk_size: int, chunk_overlap: int) -> tuple[list[BaseNode], list[tuple[int, int]]]:
    """切分一个页面文档，并同时返回节点及其原文字符区间。

    切分边界由 LlamaIndex 的 ``SentenceSplitter`` 决定；返回的偏移量是项目额外
    保存的定位信息，后续可以据此计算一个片段覆盖了哪些 PDF 物理页。
    """
    splitter = _OffsetSentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([document])
    offsets = splitter._validated_offsets(document.text, [node.get_content() for node in nodes])
    return nodes, offsets
