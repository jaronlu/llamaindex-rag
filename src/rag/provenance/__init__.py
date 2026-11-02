"""来源标识、索引清单读取、来源核验和页码展示等跨阶段规则。

这些规则同时被加载、切片、索引、检索和命令行使用；集中到 PRD 规划的
``provenance`` 目录后，各阶段不再维护容易分叉的副本。
"""

from .verify import file_sha256, format_pages, read_manifest, source_id_for, source_status

__all__ = ["file_sha256", "format_pages", "read_manifest", "source_id_for", "source_status"]
