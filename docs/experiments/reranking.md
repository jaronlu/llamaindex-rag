# 重排实验记录

## 实现协议

日期：2026-10-30。

当前实现提供两种可对照模式：

- dense 基线：直接从持久化 `VectorStoreIndex` 取最终 `top_k`；
- 两阶段检索：dense 先取 `candidate_k=20`，再由本地 CrossEncoder 重新排序并交付 `top_k=5`。

重排结果继续使用索引中保存的 Node 正文、文件名、路径和 PDF 物理页码。输出同时保留 dense 原始排名和相似度分数，以及 rerank 新排名和重排分数；两类分数不在同一尺度上，不能直接比较。模型加载使用 `local_files_only=True`，问题和文档不会发送到外部服务。

## 可复现实验命令

默认模型路径集中配置在 `src/rag/config/retrieval.py`，当前使用本机 Hugging Face 缓存中的 `BAAI/bge-reranker-v2-m3`。运行时无需传入重排模型路径；若要临时覆盖，可传 `--rerank-model-path`：

```bash
uv run python -m src.rag.cli.retrieval search data/index-full-384 \
  "加班费、年休假该怎么算？" \
  --candidate-k 20 --top-k 5
```

使用同一问题增加 `--dense-only` 并设置 `--top-k 5`，即可得到 dense 基线。实验应固定索引、嵌入模型、问题和最终输出数量，只比较是否启用重排。

## 当前状态

核心逻辑和 CLI 已通过自动化测试，覆盖候选重新排序、原始排名保留、候选不足、参数边界、本地模型路径检查以及禁止静默退回 dense。尚未在本地 CrossEncoder 模型上完成真实 PDF 的人工质量比较，因此当前不能声称重排已经改善召回或回答质量。

后续人工验证至少使用既有的三个文档内问题和一个文档外问题，记录 dense Top 5 与 dense Top 20 → rerank Top 5 的候选、来源页码、排名变化、人工判断和耗时。重排只能处理 dense 已召回的 20 条候选；如果相关片段没有进入候选集，应单独评估 hybrid 或其他召回策略。
