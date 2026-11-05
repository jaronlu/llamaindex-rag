# TODO

未实现方案的唯一跟踪处。

## 检索与评测（设计记录：docs/experiments/question-types.md）

- [ ] 查询分解：LLM 将总结型问题拆为 4~8 个子查询，多路检索后 RRF 融合（组件 `QueryFusionRetriever`）。验收：重跑“全书关于睡眠的建议”，与 dense-only 基线比较条目级覆盖度和命中页码分布。
- [ ] top_k 按问题类型配置：`questions.json` 增加 `type` 字段，事实型 5~10、总结型 20~50，命令行显式传参可覆盖。验收：输出能辨认所用配置。
- [ ] 条目级覆盖度标注与评测：总结型 gold 从 `answer_pages` 扩展为条目集合（如 `related_items`）；LLM 提议候选、人工确认。
- [ ] 问题类型枚举扩展：`type` 起步三值 `factual` / `summary` / `unanswerable`（`outside-topic` 一题可直接归入 `unanswerable`），该类考察“明确表示书内无相关内容”而非命中指标；`multi_hop` 预留，待测试集出现真实多跳失败案例再加。每新增一类，先定义 gold 标注方式与对应指标。
- [ ] 融合分口径：RRF 排序分与现有“相似度分数说明”口径不同，实现时定义如何呈现。

## 回答（设计记录：docs/experiments/question-types.md）

- [ ] tree_summarize 合成：总结型问题分批摘要再合并（组件 `TreeSummarize`）。前置取舍：递归摘要会洗掉来源，与 PRD 溯源要求冲突，需先在“引用标记嵌入文本走原生组件”与“在 `generate.py` 自实现”之间二选一。

## 索引（暂缓）

- small-to-big 分层索引：明确暂不做。重新评估条件：多文档接入后跨文档总结类问题增多，或覆盖度评测显示单查询路线上限不足。

## 其他模块遗留

- [ ] 清单页过滤：重排的假阳性来源是按时间罗列步骤的“清单页”（448–451、455 页），是否排除或按片段类型过滤留待决定。（docs/experiments/reranking.md）
- [ ] 书外问题阈值：书外问题最高重排分与文档内相差两个数量级，但程序无阈值逻辑、照常返回候选；是否加阈值需更多样本。（docs/experiments/retrieval-eval.md）
- [ ] 正文纯度：前言与项目声明混入正文检索结果，需切片或重排改善上下文纯度。（docs/experiments/retrieval.md）
