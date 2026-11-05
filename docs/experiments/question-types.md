# 问题类型与评测：事实型 vs 总结型

## 背景

现有 `questions.json` 的 8 个问题全部是事实型：答案集中在少数页面，可以标注单一答案页，用名次（Hit@k、MRR 思路）评测。2026-11-05 用两条真实查询对比两类问题后，记录差异、指标选择和 LlamaIndex 侧的处理方案，作为扩展问题集与实现查询分解前的设计依据。

## 两类问题的区别

| | 事实型 | 总结型 |
|---|---|---|
| 示例 | 书里说缓解焦虑的方法是什么？ | 全书关于睡眠的建议有哪些？ |
| 答案分布 | 集中在 1~2 个片段 | 散布全书几十个片段 |
| gold 定义 | 单个答案页或片段 | 相关条目（片段）的集合 |
| 适用指标 | Hit@k、MRR | 条目级覆盖度（Coverage@k） |
| 合成模式 | compact | tree_summarize |

单一 gold 的名次指标对总结型问题失效：答案本来就该来自几十处，“正确片段排第几”没有意义。

## 真实查询观察（2026-11-05）

条件：索引 `data/index-pages-14-292`（索引名提示为 14–292 页子集，全书共 458 页），`--dense-only --top-k 5`，未加重排。

### 总结型：“全书关于睡眠的建议有哪些？”

1. 覆盖表现：top-5 的页码只有 46、47、65、67，全部集中在相邻两节。当前条件下漏召回比例无法量化——这正说明需要条目级 gold（全书讲睡眠的建议共 N 条）才能把“覆盖不全”从观感变成数字。
2. 边界噪音：排名第 1 的片段尾部混入下一条目开头（“14. 常打网球、羽毛球、乒乓球这类球拍类运动”），排名第 4 尾部混入“4. 下午两点以后不碰咖啡因”。这是切片边界截断在检索输出中的直接体现。
3. 参考文献尾巴：排名第 5 整段是文献列表，单独作为候选信息量很低，它实际是排名第 2 片段的延续。
4. 分数窄带：5 条分数在 0.616–0.670 之间，区分度低，排序易受小扰动影响，重排与 hybrid 有明确价值。

### 事实型：“书里说缓解焦虑的方法是什么？”

结果多与抑郁、愤怒、反刍思维相关（页 69–74）。书里是否有专门的焦虑章节未被召回、排名第 1 是否恰当，目前无法判断——没有 gold 标注就无法评判检索质量。这印证了固定测试问题集的必要性：没有它，任何一次检索都只能“看观感”。

## 总结型评测：条目级覆盖度

本书正文按编号建议条目组织（如“13. 每晚睡 7 小时左右，作息固定”）。总结型问题可把“相关条目集合”作为 gold：人工整理全书讲睡眠的建议共 N 条，检索 top-k 命中其中几条即为覆盖度。比片段级标注省力，也符合“覆盖全不全”的直觉。多文档评测时按 document_id 分区统计，避免一本难的书被全局平均掩盖。

## LlamaIndex 侧处理方案（设计记录）

各项方案的实现状态不在本文维护，统一在根目录 [TODO.md](../../TODO.md) 跟踪；各类型对应的 LlamaIndex 组件与覆盖范围见 [08-question-types.md](../knowledge/08-question-types.md)；本文只记录设计依据与取舍。术语约定：TODO.md 中“未实现”指 LlamaIndex 提供了现成组件、但本项目代码尚未调用；“暂不做”是明确的设计决定，不是遗漏。

### 1. 事实型链路（现状）

现状：`src.rag.cli.retrieval` 已支持 dense 检索与重排对比；回答由 `src/rag/answering/generate.py` 自行实现（`build_messages` 组织证据、引用标记、`parse_citations` 提取引用），未使用 LlamaIndex 的 ResponseSynthesizer。该链路对事实型问题够用，本方案不改动它。

### 2. 查询分解

- 组件：`QueryFusionRetriever`，位于安装包 `.venv/lib/python3.12/site-packages/llama_index/core/retrievers/fusion_retriever.py`（已确认存在）。
- 设计：LLM 先把总结型问题拆成 4~8 个子查询（按主题维度，如睡眠时长、睡前屏幕、咖啡因、作息规律），每路检索 top-k，RRF 合并去重后取前 N 条。
- 验收方式：实现后重跑“全书关于睡眠的建议有哪些？”，与 dense-only 基线比较条目级覆盖度（命中几条编号建议）与命中页码的分布宽度。
- 预期代价：每问多一次 LLM 调用（生成子查询）加多路检索延迟；子查询质量依赖拆分 prompt 的写法。
- 开放问题：RRF 融合后的排序分与现有“相似度分数说明”口径不同，实现时须定义融合分如何向使用者呈现。

### 3. top_k 按问题类型配置

- 现状：CLI 只有单一 `--top-k`，问题集没有类型字段。
- 设计：`questions.json` 增加 `type` 字段（factual / summary）；检索与回答入口按 `type` 选择档位（事实型 5~10，总结型 20~50），命令行显式传参时仍可覆盖。
- 验收：同一问题分别以两档 top_k 运行，输出中能辨认所用配置（延续现有“输出能辨认所用参数”的验收习惯）。

### 4. tree_summarize 合成

- 实现参考：`.venv/lib/python3.12/site-packages/llama_index/core/response_synthesizers/tree_summarize.py`（`TreeSummarize` 类，已确认存在）。算法：`PromptHelper.repack` 把候选片段按上下文窗口重新打包 → 各批并行摘要（`asyncio.gather`）→ 对摘要递归调用自身，直到只剩一批时输出最终回答。
- 原生启用方式：`get_response_synthesizer(response_mode="tree_summarize", llm=...)`（`response_synthesizers/factory.py`）或 `RetrieverQueryEngine.from_args(retriever, response_mode="tree_summarize")`。
- 与本项目回答模块的核心冲突：`generate.py` 未走 LlamaIndex 合成器，其引用体系（证据编号 + `parse_citations`）依赖“证据与引用标记同批呈现”；TreeSummarize 的递归摘要会把每个片段的来源信息洗掉，若不处理，回答将无法逐条溯源，违反 PRD 的来源要求。
- 候选做法（实现时二选一）：
  1. 走原生组件：把引用标记（如 `[n]`）嵌入送入的片段文本，自定义 summary prompt 要求保留标记，最终仍用 `parse_citations` 回溯；
  2. 自行实现等价逻辑：在 `generate.py` 内对总结型问题分批摘要再合并，直接复用现有引用与来源展示体系。
- 预期代价：多次 LLM 调用（批数 × 递归层数），延迟明显高于 compact；中间摘要可能丢失细节，需人工核验覆盖度。

### 5. 分层索引 small-to-big

- 方案：每节生成摘要单独建一层索引，先查摘要定位相关节，再进节内检索条目（summary → sections 的映射关系需持久化）。
- 暂不做原因：当前仅 1 份文档、内容条目化程度高，单查询的覆盖瓶颈尚未量化；先做查询分解与条目级覆盖度评测，用数据判断是否值得改索引结构。
- 重新评估条件：多文档接入后跨文档总结类问题增多，或覆盖度评测显示单查询路线上限明显不足。

## 问题集扩展计划

- `questions.json` 增加 `type` 字段，起步三值 `factual` / `summary` / `unanswerable`（`multi_hop` 预留，待真实失败案例出现再加）；总结型问题的 gold 从 `answer_pages` 扩展为条目集合（如 `related_items`）；`unanswerable` 类考察系统是否明确表示书内无相关内容，而非命中指标。
- 标注方式：LLM 按 chunk 提议候选问题与相关条目，人工逐条确认；开发过程中的核验记录可直接作为种子。

## 未验证项

截至 2026-11-05，查询分解、top_k 类型化、`tree_summarize` 合成、条目级覆盖度标注与评测均未实现，本文仅记录设计与取舍（各项跟踪见根目录 TODO.md）。每项实现后需重跑同一组查询，与 dense-only 基线对比条目覆盖度、命中页码分布及来源可核对性。

## 复现

```bash
uv run python -m src.rag.cli.retrieval search data/index-pages-14-292 \
  "全书关于睡眠的建议有哪些？" --dense-only --top-k 5
uv run python -m src.rag.cli.retrieval search data/index-pages-14-292 \
  "书里说缓解焦虑的方法是什么？" --dense-only --top-k 5
```
