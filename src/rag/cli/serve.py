"""启动局域网 HTTP 接口，把本机索引的检索和回答能力提供给手机等客户端。

服务默认只监听 ``127.0.0.1``；需要手机访问时显式传入 ``--host 0.0.0.0``。
生成模型密钥仍只从环境变量读取，客户端不会接触密钥。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from .logging import logged_command

from ..api.app import ApiSettings, create_app
from ..config.answering import (
    DEFAULT_ANSWER_API_BASE,
    DEFAULT_ANSWER_API_KEY_ENV,
    DEFAULT_ANSWER_MODEL,
)
from ..config.api import API_TOKEN_ENV, DEFAULT_API_HOST, DEFAULT_API_PORT
from ..config.common import DEFAULT_EMBEDDING_MODEL_PATH, DEFAULT_RERANK_MODEL_PATH
from ..config.indexing import ManifestKey
from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from ..provenance import read_manifest


logger = logging.getLogger(__name__)

# 这些地址表示只监听本机；其它取值一律视为已暴露到局域网，需要提示风险。
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@logged_command
def main(argv: list[str] | None = None) -> int:
    """解析服务参数、做启动前检查，然后交给 uvicorn 运行。

    启动前先校验索引是否完整发布，避免服务已经监听却只能返回错误；端口范围、
    监听地址和令牌状态会明确打印，便于确认手机是否能访问以及是否暴露到局域网。
    """
    parser = argparse.ArgumentParser(description="启动局域网检索与回答接口")
    parser.add_argument("index", type=Path, help="索引目录")
    parser.add_argument(
        "--host",
        default=DEFAULT_API_HOST,
        help=f"监听地址（默认 {DEFAULT_API_HOST}；手机访问需改为 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"监听端口（默认 {DEFAULT_API_PORT}）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_EMBEDDING_MODEL_PATH,
        help="覆盖默认的 Hugging Face embedding 模型快照目录",
    )
    parser.add_argument(
        "--rerank-model-path",
        type=Path,
        default=DEFAULT_RERANK_MODEL_PATH,
        help="覆盖配置文件中的本地 CrossEncoder 模型目录",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_ANSWER_MODEL,
        help=f"生成模型名称（默认 {DEFAULT_ANSWER_MODEL}）",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_ANSWER_API_BASE,
        help="OpenAI 兼容服务地址",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_ANSWER_API_KEY_ENV,
        help=f"保存密钥的环境变量名（默认 {DEFAULT_ANSWER_API_KEY_ENV}）",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        print(f"启动失败: 端口必须在 1–65535 之间，当前为 {args.port}", file=sys.stderr)
        return 1
    if not args.host.strip():
        print("启动失败: 监听地址不能为空", file=sys.stderr)
        return 1

    try:
        manifest = read_manifest(args.index)
    except ValueError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1

    print(f"索引: {args.index.expanduser().resolve()}；片段数: {manifest.get(ManifestKey.CHUNK_COUNT, '未知')}")
    print(f"监听: http://{args.host}:{args.port}")
    print(f"默认请求参数: dense top {DEFAULT_CANDIDATE_K} → rerank top {DEFAULT_RETRIEVAL_TOP_K}")
    print(f"生成模型: {args.llm_model}（{args.api_base}）")
    if args.host not in _LOOPBACK_HOSTS:
        print("警告：正在监听非本机地址，同一网段的其他设备可以访问；请确认网络可信，不要暴露到公网。")
    else:
        print("提示：当前只监听本机；手机访问需使用 --host 0.0.0.0 重新启动。")
    if os.environ.get(API_TOKEN_ENV, "").strip():
        print(f"访问令牌: 已启用（环境变量 {API_TOKEN_ENV}），除健康检查外的请求需要 Bearer Token。")
    else:
        print(f"访问令牌: 未设置；建议仅在可信局域网自用，或设置 {API_TOKEN_ENV} 后重启。")

    settings = ApiSettings(
        index=args.index,
        embed_model=args.model_path,
        rerank_model=args.rerank_model_path,
        answer_model=args.llm_model,
        answer_api_base=args.api_base,
        answer_api_key_env=args.api_key_env,
    )
    app = create_app(settings)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        # uvicorn 在部分平台上会把 Ctrl-C 继续抛给调用方；正常停止不应记为失败。
        print("\n服务已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
