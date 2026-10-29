"""构建本地向量索引并查看保存的片段。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from ..chunking.split import chunk_documents
from ..config.chunking import DEFAULT_CHUNK_OVERLAP
from ..config.indexing import (DEFAULT_EMBEDDING_DEVICE, DEFAULT_INDEX_CHUNK_SIZE, DEFAULT_MODEL, DEFAULT_REVISION,
                               HASH_READ_SIZE, MANIFEST_FILE, IndexCommand, ManifestKey)
from ..config.ingestion import DEFAULT_START_PAGE, MetadataKey, PageStatus
from ..indexing.store import IndexBuildError, build_index, read_node
from ..indexing.model import snapshot_identity
from ..ingestion.pdf import load_pdf


def file_hash(path: Path) -> str:
    """计算文件内容版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_READ_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """运行构建或片段回查命令。"""
    parser = argparse.ArgumentParser(description="本地索引构建和片段回查")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser(IndexCommand.BUILD)
    build.add_argument("index", type=Path)
    build.add_argument("files", type=Path, nargs="+")
    build.add_argument("--start", type=int, default=DEFAULT_START_PAGE)
    build.add_argument("--end", type=int)
    build.add_argument("--chunk-size", type=int, default=DEFAULT_INDEX_CHUNK_SIZE)
    build.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    build.add_argument("--model-path", type=Path, required=True, help=f"Hugging Face 缓存快照目录；示例 {DEFAULT_MODEL} @ {DEFAULT_REVISION}")
    overview = sub.add_parser(IndexCommand.OVERVIEW, help="查看索引中所有文档的文本覆盖情况")
    overview.add_argument("index", type=Path)
    read = sub.add_parser(IndexCommand.READ)
    read.add_argument("index", type=Path)
    read.add_argument("node_id")
    args = parser.parse_args(argv)
    # 切片前无法确定待处理数量；发布后即使输出失败也不能报告为未发布。
    node_count: int | None = None
    published = False
    try:
        if args.command == IndexCommand.OVERVIEW:
            if not args.index.exists():
                raise ValueError(f"索引不存在: {args.index}")
            manifest = json.loads((args.index / MANIFEST_FILE).read_text(encoding="utf-8"))
            if manifest.get(ManifestKey.COMPLETE) is not True:
                raise ValueError("索引不完整，无法查看")
            print(f"索引: {args.index}；片段数: {manifest.get(ManifestKey.CHUNK_COUNT, '未知')}")
            print(f"模型: {manifest.get(ManifestKey.MODEL, '未知')} @ {manifest.get(ManifestKey.VERSION, '未知')}；维度: {manifest.get(ManifestKey.DIMENSION, '未知')}；切片大小: {manifest.get(ManifestKey.CHUNK_SIZE, '未知')}；重叠: {manifest.get(ManifestKey.CHUNK_OVERLAP, '未知')}")
            for document in manifest.get(ManifestKey.DOCUMENTS, []):
                coverage = "完整" if document.get(ManifestKey.TEXT_COVERAGE_COMPLETE) is True else "不完整或未知"
                print(f"文件: {document.get(ManifestKey.PATH, '缺失')} | 物理页: {document.get(ManifestKey.START, '?')}–{document.get(ManifestKey.END, '?')} | 文本覆盖: {coverage} | 空文本页: {document.get(ManifestKey.EMPTY_PAGES, '未知')}")
            return 0
        if args.command == IndexCommand.READ:
            node = read_node(args.index, args.node_id)
            manifest = json.loads((args.index / MANIFEST_FILE).read_text(encoding="utf-8"))
            source_path = node.metadata.get(MetadataKey.SOURCE_PATH)
            record = next((item for item in manifest.get(ManifestKey.DOCUMENTS, []) if item.get(ManifestKey.PATH) == source_path), None)
            if not isinstance(source_path, str) or not source_path.strip() or not record:
                print("警告：索引中缺少该片段的来源版本记录")
            elif not Path(source_path).is_file():
                print(f"警告：来源文件不存在或已移动: {source_path}")
            elif file_hash(Path(source_path)) != record.get(ManifestKey.SHA256):
                print(f"警告：来源文件内容已变化: {source_path}")
            if record and record.get(ManifestKey.EMPTY_PAGES):
                print(f"警告：所选范围存在空文本页，文本覆盖不完整: {record[ManifestKey.EMPTY_PAGES]}")
            print(f"片段 ID: {node.node_id}\n来源: {json.dumps(node.metadata, ensure_ascii=False)}\n{node.get_content(metadata_mode=MetadataMode.NONE)}")
            return 0
        if args.index.exists():
            raise ValueError("索引位置已存在；请使用新目录进行受控重建")
        paths = [path.expanduser().resolve(strict=True) for path in args.files]
        if len(paths) != len(set(paths)):
            raise ValueError("输入包含重复文档")
        # 每个 PDF 先逐页加载，再汇总成功页面统一切片。
        documents: list[Document] = []
        versions = []
        for path in paths:
            # 读取前后各算一次摘要，避免将读取中变化的文件误记为同一版本。
            digest = file_hash(path)
            result = load_pdf(path, args.start, args.end)
            failures = [page for page in result.pages if page.status == PageStatus.FAILED]
            if failures:
                details = "; ".join(f"第 {page.number} 页: {page.error or '原因未知'}" for page in failures)
                raise ValueError(f"{path} 页面解析失败: {details}")
            empty = [page.number for page in result.pages if page.status == PageStatus.EMPTY]
            if empty:
                print(f"{path} 空文本页: {empty}")
            documents.extend(page.document for page in result.pages if page.status == PageStatus.SUCCESS and page.document is not None)
            if digest != file_hash(path):
                raise ValueError(f"读取期间文件发生变化: {path}")
            versions.append({ManifestKey.PATH: str(path), ManifestKey.SHA256: digest, ManifestKey.START: args.start, ManifestKey.END: args.end or result.total_pages, ManifestKey.EMPTY_PAGES: empty, ManifestKey.TEXT_COVERAGE_COMPLETE: not empty})
        # Document 是页面；Node 是切片。索引保存的是这些 Node。
        nodes = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
        node_count = len(nodes)
        if not nodes:
            raise ValueError("没有可索引的片段")
        model_path = args.model_path.expanduser().resolve(strict=True)
        identity = snapshot_identity(model_path)
        for item in versions:
            item[ManifestKey.SOURCE_ID] = hashlib.sha256(item[ManifestKey.PATH].encode("utf-8")).hexdigest()
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]
        model = HuggingFaceEmbedding(model_name=str(model_path), device=DEFAULT_EMBEDDING_DEVICE, local_files_only=True)
        # LlamaIndex 当前未公开底层 tokenizer；使用锁定版本的内部模型，
        # 同时计入文本前缀和特殊 token，避免预检通过后推理静默截断。
        tokenizer = model._model.tokenizer
        maximum = model.max_length
        if maximum is None:
            raise ValueError("无法确定模型的 token 长度上限")
        text_prompt = model._model.prompts.get("text") or ""
        for node in nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            if len(tokenizer.encode(text_prompt + text, add_special_tokens=True, truncation=False, verbose=False)) > maximum:
                raise ValueError(f"片段 {node.node_id} 超出模型 {maximum} token 上限；请减小切片大小后重试")
        if snapshot_identity(model_path) != identity:
            raise ValueError("模型快照加载期间发生变化，请重新运行")
        manifest = build_index(nodes, model, args.index, {**identity, ManifestKey.CHUNK_SIZE: args.chunk_size, ManifestKey.CHUNK_OVERLAP: args.chunk_overlap, ManifestKey.DOCUMENTS: versions})
        published = True
        print(f"索引完成: {args.index.resolve()}\n模型: {identity[ManifestKey.MODEL]} @ {identity[ManifestKey.VERSION]}\n维度: {manifest[ManifestKey.DIMENSION]}；成功: {len(nodes)}；失败: 0")
        for node in nodes:
            print(f"片段 ID: {node.node_id} | 文件: {node.metadata.get(MetadataKey.SOURCE_PATH)} | 物理页: {node.metadata.get(MetadataKey.PAGE_NUMBERS)}")
        return 0
    except Exception as exc:
        if args.command == IndexCommand.BUILD and published:
            print(f"索引已发布，但输出失败: {exc}", file=sys.stderr)
        elif args.command == IndexCommand.BUILD and not isinstance(exc, IndexBuildError):
            remaining = node_count if node_count is not None else "未知"
            print(f"索引构建失败：向量化成功 0，向量化失败 0，未处理 {remaining}；本次索引未发布。原因: {exc}", file=sys.stderr)
        else:
            print(f"索引操作失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
