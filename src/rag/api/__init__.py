"""局域网 HTTP 接口，把检索、重排和回答能力提供给同网段的客户端。

接口层只做参数校验、鉴权和序列化；索引兼容性、来源核验和生成模型调用沿用
``retrieval`` 与 ``answering`` 模块，避免在 HTTP 层复制一套 RAG 规则。
"""

from .app import ApiSettings, create_app

__all__ = ["ApiSettings", "create_app"]
