"""命令行共享的回答、引用和证据展示。

回答命令和连续对话命令都必须用同一套措辞展示引用检查和来源状态；放在这里
避免两处各写一份，导致同一个回答在一个入口提示“不可核对”、在另一个入口
看起来正常。这里只负责输出，不重新检索或判断回答是否正确。
"""

from __future__ import annotations

from collections.abc import Sequence

from ..answering.generate import Answer
from ..retrieval.search import RetrievalResult


def print_citation_check(answer: Answer) -> None:
    """输出引用编号和来源状态检查，明确区分可核对和不可核对的情况。"""
    if answer.invalid_citations:
        labels = ", ".join(f"[{number}]" for number in answer.invalid_citations)
        print(f"\n引用检查: 回答包含不存在的证据编号 {labels}；这些编号无法对应原文，不能作为依据。")
    elif answer.citations:
        labels = ", ".join(f"[{number}]" for number in answer.citations)
        print(f"\n引用检查: 回答引用了 {labels}。")
    else:
        print("\n引用检查: 回答没有出现 [n] 引用。若结论是资料不足，请人工确认是否确实缺乏依据；若是正常回答，则无法把陈述对应到原文。")

    # 被引用的证据仍可能因源文件缺失或版本变化而无法核对；不单独提示的话，
    # 读者容易把“有页码”误当成“已验证”。
    cited = set(answer.citations) - set(answer.invalid_citations)
    unverified = [number for number in sorted(cited) if not answer.evidence[number - 1].provenance_complete]
    if unverified:
        labels = ", ".join(f"[{number}]" for number in unverified)
        print(f"来源检查: 引用的证据 {labels} 无法回到当前文件核对；回答中的相应内容不能视为已验证。")


def print_evidence(
    evidence: Sequence[RetrievalResult],
    cited: set[int],
    *,
    show_text: bool = True,
) -> None:
    """打印每条证据的来源信息，并按需展示完整正文。

    show_text 为假时只显示文件名、页码和来源状态，适合连续对话中保持输出简洁；
    完整正文仍保存在索引和证据对象里，可用 --show-evidence 或单次回答命令查看。
    """
    for number, result in enumerate(evidence, start=1):
        marker = "已被引用" if number in cited else "未被引用"
        print(f"\n--- 证据 {number}（{marker}）---")
        print(f"文件名: {result.file_name}")
        print(f"来源路径: {result.source_path}")
        print(f"物理页码: {result.page_label}")
        print(f"片段 ID: {result.node_id}")
        print(f"来源状态: {result.source_status_label}")
        if show_text:
            print("完整原文:")
            print(result.text)
