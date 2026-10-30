"""检索阶段的本机模型和候选数量默认配置。"""

import os
from pathlib import Path

DEFAULT_RERANK_MODEL_PATH = Path(
    os.environ.get(
        "RAG_RERANK_MODEL_PATH",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        ),
    )
).expanduser()
DEFAULT_CANDIDATE_K = 20
DEFAULT_RETRIEVAL_TOP_K = 5
