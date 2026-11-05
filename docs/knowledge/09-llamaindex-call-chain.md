# LlamaIndex 调用链条

记录日期：2026-11-05。以下以项目锁定版本 `llama-index-core 0.14.25`、`llama-index-embeddings-huggingface 0.6.1`、`llama-index-llms-openai-like 0.8.1` 为准；内部 API 在升级后可能变化，需重新核对。

## 总览

项目没有使用 `QueryEngine` 一次完成“检索 + 合成”，而是直接调用索引和检索器，自己拼装证据与提示词。这样做的目的是让每一段正文、页码和引用编号都能在交付前被检查和改写。

```text
CLI / FastAPI
  │
  ├─ ingestion.pdf.load_pdf
  │    pypdf 提取文本 → llama_index.core.Document（每页一个）
  │
  ├─ chunking.split.chunk_documents
  │    _OffsetSentenceSplitter(SentenceSplitter).get_nodes_from_documents
  │    → TextNode（正文 + 页码 + 字符偏移 + 相邻节点关系）
  │
  ├─ indexing.store.build_index
  │    HuggingFaceEmbedding.get_text_embedding（逐片）
  │    → VectorStoreIndex(prepared, embed_model).storage_context.persist
  │
  ├─ retrieval.search.retrieve
  │    StorageContext.from_defaults → load_index_from_storage
  │    → index.as_retriever(similarity_top_k) → retriever.retrieve(question)
  │    → NodeWithScore → RetrievalResult（来源校验）
  │
  ├─ retrieval.search.rerank
  │    sentence_transformers.CrossEncoder（框架外）
  │
  └─ answering.generate.generate_answer
       ChatMessage + OpenAILike.chat / stream_chat → Answer（引用编号检查）
```

## 分阶段调用

| 阶段 | 项目入口 | 调用的 LlamaIndex 组件 | 项目额外封装 |
| --- | --- | --- | --- |
| 加载 | `cli/loading.py` → `ingestion/pdf.py:load_pdf` | `Document`、`BaseReader` | 用 `pypdf` 逐页取文本；记录每页成功、空文本或失败；写入 `file_name`、`source_path`、`source_id`、`page_numbers`。 |
| 切片 | `cli/chunking.py`、`cli/indexing.py` → `chunking/split.py:chunk_documents` | `SentenceSplitter`、`TextNode`、`NodeRelationship` | 合并同文件连续页；校验来源字段；按结构条目预切；记录原文字符偏移并换算物理页；建立 `PREVIOUS` / `NEXT` 关系。 |
| 向量化 | `cli/indexing.py` build → `indexing/store.py:build_index` | `HuggingFaceEmbedding`、`BaseEmbedding` | 从本地快照加载模型，禁止联网；逐片调用 `get_text_embedding`；校验空向量、非有限值和维度一致；只嵌入正文，不嵌入来源 metadata。 |
| 建索引 | 同上 | `VectorStoreIndex`、`StorageContext.persist` | 先写入临时目录，再写项目自己的 `manifest.json`，最后原子发布；拒绝覆盖已有目录和重复 `node_id`。 |
| 读索引 | `indexing/store.py:read_node` | `StorageContext.from_defaults`、`docstore.get_node` | 先检查 manifest 完成标记，再按 `node_id` 回查片段正文和来源。 |
| 检索 | `retrieval/search.py:retrieve` | `load_index_from_storage`、`index.as_retriever`、`retriever.retrieve` | 校验模型身份与索引兼容性；把 `NodeWithScore` 转成 `RetrievalResult`；过滤来源或正文不合法的节点。 |
| 重排 | `retrieval/search.py:rerank` | 无（直接用 `sentence_transformers.CrossEncoder`） | 保留 dense 分数、原始排名和重排分数三个字段；重排模型也要求本地目录。 |
| 回答 | `answering/generate.py:generate_answer` | `ChatMessage`、`MessageRole`、`OpenAILike.chat` / `stream_chat` | 按相关度阈值筛选并重新编号证据；拼装“文件 + 物理页码 + 完整正文”的提示词；解析 `[n]` 并检查越界；识别空回答和 `finish_reason=length`。 |
| 连续对话 | `cli/chat.py` → `answering/conversation.py:condense_question` | 同一个 `ChatModel` 接口（生产为 `OpenAILike`） | 用最近几轮问答把追问改写成独立问题；改写失败回退原问题；检索阶段复用 `RetrievalSession` 已加载的索引、embedding 和重排模型。 |

## 关键调用细节

### 1. 加载：Document 是页，不是文件

`PagePDFReader` 继承 `BaseReader`，但命令行实际调用的是项目新增的 `load_pages`，而不是 `load_data`。原因是 `load_data` 遇到任一页解析失败就整份文件抛错，而分阶段学习需要看到每一页的状态。每页生成一个 `Document`，页码放在 metadata 中，后续切片和溯源都依赖它。

### 2. 切片：结构边界由项目划，句子合并交给框架

`_OffsetSentenceSplitter` 继承并复用 `SentenceSplitter`：

- `_split` 先按条目或项目符号切出结构块；块不超限时整块保留，超限才回退到父类按句子切分。
- `_merge` 复刻父类的 token 合并与重叠逻辑，同时记录每个输出片段在拼接文本中的左闭右开字符区间。
- `_validated_offsets` 用原文切片反查，确认片段和偏移一一对应；对不上就报错，避免来源页码错位。

合并连续页后，项目用字符区间与各页区间求交，得到跨页片段实际覆盖的所有物理页。节点 ID、正文和 metadata 由 LlamaIndex 的 `TextNode` 承载。

### 3. 索引：显式逐片嵌入，再交给 VectorStoreIndex

`build_index` 没有依赖 `VectorStoreIndex.from_documents` 的隐式嵌入流程，而是：

1. 对每个节点副本调用 `embed_model.get_text_embedding(node.get_content(metadata_mode=MetadataMode.NONE))`；
2. 把向量写回 `copy.embedding`；
3. 用已带向量的节点构造 `VectorStoreIndex(prepared, embed_model=embed_model)`；
4. 调用 `index.storage_context.persist(persist_dir=staging)` 持久化。

这样可以在嵌入前后逐个节点校验长度、维度和失败数量，也确保来源 metadata 不会被送进嵌入模型。默认向量存储是 `SimpleVectorStore`，落盘为索引目录中的 JSON 文件。

### 4. 检索：直接使用 Retriever，不经过 QueryEngine

`_prepare_retriever` 的顺序是：

```python
context = StorageContext.from_defaults(persist_dir=str(index_path))
index = load_index_from_storage(context, embed_model=model)
retriever = index.as_retriever(similarity_top_k=top_k, embed_model=model)
raw_results = retriever.retrieve(question)  # list[NodeWithScore]
```

项目只取 `node`、`node.metadata`、`node.get_content(MetadataMode.NONE)` 和 `score`，再自行校验来源。检索过程不会重新读取 PDF，正文和页码都来自持久化节点。

### 5. 回答：ChatMessage + OpenAILike，而不是 ResponseSynthesizer

`build_messages` 生成一条 `MessageRole.SYSTEM` 消息和一条 `MessageRole.USER` 消息，后者包含编号后的证据正文。非流式调用 `llm.chat(messages)` 后读取 `ChatResponse.message.content`；流式调用 `llm.stream_chat(messages)` 并逐段读取 `chunk.delta`。两种方式最终都进入同一个 `_answer_from_content`，保证引用检查规则一致。

跳过 `ResponseSynthesizer` 的原因不是框架能力不足，而是项目需要先把“哪些片段作为证据、编号如何对应、来源能否核对”固定下来，再让模型生成；框架的默认合成流程会把这些中间状态封装掉。

## 框架内与框架外的职责边界

| 由 LlamaIndex 负责 | 由本项目负责 |
| --- | --- |
| `Document` / `TextNode` 数据结构和序列化 | PDF 文本提取、逐页状态、来源身份与物理页码 |
| 句子级切分、token 计数与重叠合并 | 连续页分组、结构条目边界、字符偏移校验、跨页归属 |
| 嵌入模型调用接口与向量封装 | 本地快照策略、逐片检查、维度与失败统计 |
| 向量索引、docstore、持久化文件格式 | 临时目录构建、manifest、原子发布、不覆盖旧索引 |
| 相似度检索与 `NodeWithScore` | 来源过滤、结果转换、兼容性检查、人可读输出 |
| LLM 客户端协议与消息类型 | 提示词、证据编号、相关度阈值、引用检查、错误提示 |

## 未使用的框架路径

以下组件在当前代码中没有接入，记录在此避免误以为已经使用：

- `QueryEngine` / `RetrieverQueryEngine` / `ResponseSynthesizer`：检索和合成被拆开，项目自己控制证据与引用。
- `IngestionPipeline`：加载、切片、嵌入分属独立命令，便于逐阶段人工检查。
- `SimpleDirectoryReader`、`llama-index-readers-file`：PDF 由 `pypdf` 直接逐页读取，以获得页面级状态。
- `NodePostprocessor`：重排直接调用 `CrossEncoder`，尚未接入框架的后处理链。
- `Settings` 全局配置：模型和设备通过函数参数及 `config/` 显式传入，避免隐式全局状态。

## 升级时需要重新验证的内部依赖

- `SentenceSplitter._split` / `_merge` 的签名和 `_Split` 结构：`chunking/offset_splitter.py` 直接继承并覆盖了这两个内部方法。
- `HuggingFaceEmbedding._model.tokenizer`、`max_length`、`_model.prompts`：索引构建前的 token 上限预检依赖这些内部字段。
- `StorageContext.persist` 的落盘文件布局：manifest 之外的索引文件由框架管理，升级后应核对能否用 `load_index_from_storage` 正常读回。
- `NodeWithScore` 的 `node` / `score` 属性，以及 `ChatResponse` 的 `message.content`、`delta`、`raw` 结构：检索和回答的转换层直接读取这些字段。

升级后至少重跑一次“加载 → 切片 → 构建索引 → 检索 → 回答”的真实 PDF 流程，并检查页码、片段 ID 和引用编号是否仍然对应。

## 相关文档

- 各阶段能力与限制：[加载](03-load-issues.md)、[切片](04-chunk-issues.md)、[索引](05-indexing-issues.md)、[重排](06-rerank-issues.md)、[回答](07-answer-issues.md)
- 切片组件对比：[LlamaIndex 切片能力](01-llamaindex-chunking.md)
- 运行命令与安装说明：[README](../../README.md)
