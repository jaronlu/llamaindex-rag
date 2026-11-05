# 问题类型与 LlamaIndex 组件覆盖图谱

记录日期：2026-11-05。文中组件均在项目安装的 `llama-index-core`（`.venv/lib/python3.12/site-packages/llama_index/core/`）中用 grep 确认存在，但未实跑；“有组件”不等于“已接入本项目”，接入状态统一见根目录 [TODO](../../TODO.md)。

## 为什么需要类型维度

不同类型的问题在检索策略、合成方式、gold 标注和评测指标上都不同。“事实型 / 总结型”只是**答案分布**这一个轴的两端，完整维度还有跳数、可答性和答案形态。

## 类型维度（通用）与分布（领域相关）

| 维度轴 | 本质 | 换领域的例子 |
| --- | --- | --- |
| 答案分布 | 集中（单段）vs 分散（全书） | 合同库：“违约怎么定义” vs “哪些条款保护乙方” |
| 跳数 | 一段证据 vs 多段语义不相似片段组合 | 代码库 RAG 跨调用链找报错原因 |
| 可答性 | 语料内 vs 语料外 | 任何知识库都有“库里没有”的问题 |
| 答案形态 | 定位 / 列举 / 对比 / 流程 / 计数 | 财报 RAG 的跨表聚合 |

分类维度是问答领域积累的通用框架（如 SQuAD 2.0 的不可答、HotpotQA 的多跳）；**类型分布由语料结构和使用场景决定，gold 标注方式与指标由本项目语料决定**。例如本书正文按编号建议条目组织，才有“条目级覆盖度”这种低标注成本的指标；散文体语料只能退回片段级 coverage。本书正文存在交叉引用（如片段中出现“熬夜之后怎么补见本节第 38 条”），多跳型问题因此可能出现；引用密度未做统计。

## 类型 × 组件覆盖图谱

| 类型 | LlamaIndex 覆盖 | 组件（安装包路径已确认） | 现实提醒 |
| --- | --- | --- | --- |
| 事实型（单跳） | 有，核心路径 | `VectorStoreIndex.as_retriever` + QueryEngine | 本项目现状，够用 |
| 总结型 | 有，多种解 | `TreeSummarize`（`response_synthesizers/tree_summarize.py`，分治式：repack → 各批并行摘要 → 递归合并）；`SummaryIndex`（`indices/list/base.py`）；`DocumentSummaryIndex`（`indices/document_summary/base.py`，每文档建摘要层，专为全书之问设计） | 递归摘要会洗掉片段来源，接入须解决与溯源要求的冲突（取舍记录见 question-types.md） |
| 多跳型 | 有，一等公民 | `SubQuestionQueryEngine`（`query_engine/sub_question_query_engine.py`）：LLM 拆子问题 → 逐个子引擎作答 → 汇总 | 每个子问题一次完整 LLM 调用，慢且贵；子问题质量依赖拆分 prompt |
| 比较型 | 有 | 同 `SubQuestionQueryEngine`，常用于两文档对比场景 | 拆成两路各自检索再对比，天然契合 |
| 流程型（顺序敏感） | 无专门组件 | 靠切片策略（一个流程不切散）、元数据记步骤号、合成 prompt 要求保序 | 框架只给挂点，逻辑自行组装 |
| 不可答 | 无内置拒绝机制 | 只有原料：书内外检索分数差、自定义合成 prompt（“上下文不足就明说”）、结构化输出加 `has_answer` 字段 | 本项目已有分数差实证（2026-10-30 重排评测、`index-from14-384`：书外最高 0.0065，书内 0.78–0.97；注意 11-05 的 dense 运行未加重排，分数不可混用），阈值方案在 TODO |
| 聚合 / 计数 | 基本无解 | 出路在框架外：属性抽成元数据后用 `MetadataFilters` 过滤再本地统计，或走 text-to-SQL | 不要指望合成器“数数”；本项目明确记为不支持 |
| 元问题（问文档本身） | 半个 | `DocumentSummaryIndex` 能答“这本书讲什么”；“证据等级体系是什么”取决于属性抽取 | 半 DIY |

## 设计哲学：组合件而非一类一组件

`RouterQueryEngine`（`query_engine/router_query_engine.py`）用 LLM 按问题类型把请求分发到不同引擎（普通 / tree_summarize / sub_question）。这说明 LlamaIndex 的预期用法是：router 分发 + 各类型各选组件，而非单一管道适配所有问题。

对本项目的含义：

- 已规划的查询分解（`QueryFusionRetriever`）覆盖总结、多跳、比较的**检索层**；`tree_summarize` 覆盖总结的**合成层**。
- “top_k 按问题类型配置”的候选升级方向是 router 分发：参数化范围从 top_k 扩大到引擎选择。是否升级待条目级覆盖度基线出来后再定。
- 聚合型、流程型不进当前测试集，先在问题集中保留 `unanswerable` 观察拒答能力。

## 类型落地约定

- 问题集 `type` 起步三值：`factual` / `summary` / `unanswerable`；`multi_hop` 预留，待测试集出现真实多跳失败案例再加。
- 每新增一类，先回答两个问题：gold 怎么标？指标换成什么？答不出就不进测试集。
- 分类框架的用途是查漏：某类问题反复失败时，先判断是否属于新类型，再决定是检索质量问题还是管道能力缺失。

## 相关文档

- 设计决策与实验观察：[question-types.md](../experiments/question-types.md)
- 实现状态跟踪：根目录 [TODO.md](../../TODO.md)
- 检索评测基线与问题集：[retrieval-eval.md](../experiments/retrieval-eval.md)、[questions.json](../experiments/questions.json)
