# LlamaIndex 切片能力

以下组件以项目安装的 `llama-index-core 0.14.25` 为参考。具体 API、可选集成和依赖以实际安装版本为准。

| 组件 | 主要思路 | 适合的场景 |
| --- | --- | --- |
| `SentenceSplitter` | 优先沿段落、句子边界切分，再按 token 大小组合，可配置重叠 | 普通文章、书籍正文；当前项目使用的方案 |
| `TokenTextSplitter` | 主要按 token 数量限制切分，可配置重叠 | 需要更直接地控制片段大小；可能在句子中间切开 |
| `SentenceWindowNodeParser` | 以单个句子创建节点，同时把周围句子窗口放入元数据 | 希望按细粒度句子检索，再把周围上下文提供给后续阶段 |
| `SemanticSplitterNodeParser` | 使用嵌入模型比较相邻句子的语义，在语义变化处切分 | 主题边界比固定长度更重要的长文本；需要嵌入模型 |
| `SemanticDoubleMergingSplitterNodeParser` | 基于句子窗口和语义相似度进行合并 | 探索更复杂的语义分段；需要结合具体版本配置与验证 |
| `HierarchicalNodeParser` | 按多个粒度生成父子节点，例如大段、中段、小段 | 检索时使用细粒度节点，同时需要找回更大的上下文 |
| `MarkdownNodeParser` | 利用 Markdown 标题等结构切分 | 已有结构清晰的 Markdown 文档 |
| `HTMLNodeParser` | 按 HTML 标签和结构解析 | 网页或 HTML 文档 |
| `JSONNodeParser` | 利用 JSON 结构生成节点 | 配置、记录等结构化 JSON |
| `CodeSplitter` | 根据编程语言的代码结构切分 | 源码或代码文档；具体语言和解析依赖需核对 |
| `MarkdownElementNodeParser` | 解析 Markdown 中的元素，包括更复杂的结构 | 需要单独处理文档元素时；配置和依赖需按版本确认 |
| `SimpleFileNodeParser` | 根据文件类型选择相应解析器 | 混合格式文档的基础解析 |

`NodeParser` 负责把 `Document` 转成 `Node`，还可以携带元数据和节点关系；`TextSplitter` 则是偏向文本切分的一类解析器。

当前 PDF 流程拼接同一文件的连续物理页后切片，并记录片段覆盖的页码；空页和缺页处断开。更换切片器本身不能解决来源映射问题。语义切片需要嵌入模型；使用外部服务前应确认数据流向、配置与成本。
