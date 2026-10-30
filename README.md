# LlamaIndex RAG

## 从哪里开始读代码

目前已实现到检索阶段：`PDF → 页面 Document → 切片 Node → 向量和持久化索引 → dense 检索`，并提供可选的 `dense top 20 → rerank top 5` 两阶段检索。基于来源的回答尚未实现；检索分数和重排分数都不能单独证明候选相关或答案正确。

1. 从 [加载命令](src/rag/cli/loading.py) 开始，看参数如何传给 [PDF 加载器](src/rag/ingestion/pdf.py)。每个页面的正文和来源信息放在 LlamaIndex `Document` 中；物理页码从 1 开始。
2. 接着看 [切片命令](src/rag/cli/chunking.py) 和 [切片逻辑](src/rag/chunking/split.py)。`Node` 是可索引的片段，继承文件和页码信息；连续页面可以组成跨页片段。[偏移量切分器](src/rag/chunking/offset_splitter.py) 记录片段在拼接原文中的字符位置，以确定覆盖了哪些页面。
3. 接着看 [索引命令](src/rag/cli/indexing.py) 和 [索引存储](src/rag/indexing/store.py)。嵌入模型将每个片段的正文转换成一组数字（向量），`VectorStoreIndex` 保存向量和节点；`manifest.json` 记录构建配置与文档版本。之后可用节点 ID 从已保存的文档库找回原文和来源。
4. 再看 [检索命令](src/rag/cli/retrieval.py) 和 [检索逻辑](src/rag/retrieval/search.py)。检索先核对模型快照和索引配置，再让 LlamaIndex 按 Top-K 查询；可选的本地 CrossEncoder 会对 dense 候选重新排序，结果继续使用索引中保存的完整正文与来源元数据。

`config/` 集中存放默认值和字段名；`tests/` 展示各阶段的输入、边界和预期。命令行模块负责接收参数及展示结果，阶段目录负责实际处理。下面的命令可以边运行边对照这些文件阅读。

## 安装依赖

需要安装 [uv](https://docs.astral.sh/uv/) 和 Python 3.12。在项目根目录运行：

```bash
uv sync
```

验证环境：

```bash
uv run python -c 'from llama_index.core import SimpleDirectoryReader; from llama_index.readers.file import PDFReader; print("依赖导入成功")'
```

依赖版本记录在 `uv.lock`。

## 运行日志

四个阶段命令行入口会将阶段开始、结束、耗时、参数、数量和失败信息写入 `data/logs/rag.log`。每次命令有独立的 `run_id`，可用它串联加载、切片、索引与检索事件及异常堆栈的每一行；非零退出码以 `WARNING command.finished` 标记；索引日志还记录向量化数量和发布状态。终端继续显示供人工核对的完整正文和来源。日志默认不主动记录正文、切片或向量，但文件路径和第三方异常及堆栈可能包含敏感内容，请按本地资料的保密要求管理日志。

可通过环境变量配置日志文件和级别（`DEBUG`、`INFO`、`WARNING`、`ERROR`）：

```bash
RAG_LOG_FILE=data/logs/debug.log RAG_LOG_LEVEL=DEBUG uv run python -m src.rag.cli.loading docs/pdf/HowToLiveBetter.pdf --start 2 --end 3
```

默认级别为 `INFO`；日志达到 5 MiB 时轮转，保留 3 个旧文件。`data/` 已被 Git 忽略。无法创建或写入日志时会在标准错误提示，业务操作继续执行；日志可能不完整。当前文件轮转面向单进程命令，不保证多个进程同时写同一文件时的完整性；并发运行时请为每个进程设置不同的 `RAG_LOG_FILE`。自动化测试使用临时日志，不写入日常日志；日志只覆盖项目显式记录的事件，并不等同于 LlamaIndex 内部完整调用追踪。作为库直接调用阶段函数时，日志由调用方配置。当前阶段均在命令所在的线程执行；未来若将阶段任务交给线程，须在提交每个任务时使用 `src.rag.cli.logging.bind_log_context` 包装任务，并在命令退出前等待任务结束，才能关联运行 ID。

## 文档加载

在项目根目录运行以下命令，查看测试文档的第 2 至 3 个物理页面（页码从 1 开始，包含结束页）：

```bash
uv run python -m src.rag.cli.loading docs/pdf/HowToLiveBetter.pdf --start 2 --end 3
```

省略页码参数可检查全文；替换路径即可加载其他 PDF。输出包含完整页面文本、文件路径、物理页码和成功、空文本、失败页统计。对照 PDF 检查中文和段落顺序；空文本页需要人工确认。失败页不影响其他页面处理，但命令返回非零状态。暂不支持 OCR，扫描件或图片中的文字可能无法提取。

```bash
uv run python -m unittest discover -s tests -v
```

本模块使用 LlamaIndex `BaseReader` 和 `Document`，通过 `pypdf` 逐页提取文本。这样可以选择页码范围并报告单页失败；`Document.metadata` 保留文件路径、文件标识及物理页码。文件标识由绝对路径生成，移动文件后标识会变化。

## 文本切片

在项目根目录运行，查看第 5 页的完整切片及相邻片段：

```bash
uv run python -m src.rag.cli.chunking docs/pdf/HowToLiveBetter.pdf --start 5 --end 5 --chunk-size 128 --chunk-overlap 32
```

将参数改为 `--chunk-size 256 --chunk-overlap 64`，可对照同一页的边界。省略 `--end` 会处理到末页；替换路径可检查其他 PDF。默认大小为 512、重叠为 64，单位是 LlamaIndex 默认 tokenizer 的 token，长度统计针对片段正文；重叠为上限，实际取决于句子边界。配置须满足大小大于 0，且 `0 ≤ 重叠 < 大小`。输出包含完整文本、片段编号、运行时节点 ID、文件路径、物理页码及长度统计；节点 ID 每次运行会重新生成。空页和失败页会明确列出，失败页使命令返回非零状态。

使用 LlamaIndex `SentenceSplitter` 的句子拆分规则切片，通过适配其内部合并步骤记录原文位置；这依赖当前锁定版本的内部接口，升级 LlamaIndex 时应重新验证。同一文件的连续物理页面以换行拼接，切片可跨页；PDF 的 `page_numbers` 是唯一页码来源，单页例如 `[5]`，跨页例如 `[5, 6]`，命令行由它生成位置展示。缺页、空页及不同文件之间不拼接。切片入口目前支持 PDF，要求文件名、来源路径、来源标识及物理页码；其他格式引入时再设计其定位方式。切片大小仅计算正文，来源元数据仍保存在节点中。重叠是最大值，并不保证每个边界或每个观点完整；PDF 提取的页眉、页脚和换行可能干扰阅读，应对照原 PDF。此阶段只做本地处理，不调用模型或外部服务。实验结果和限制见 [切片实验记录](docs/experiments/chunking.md)。

## 向量化与索引

本阶段使用本地 `BAAI/bge-small-zh-v1.5` 和 LlamaIndex `HuggingFaceEmbedding`、`VectorStoreIndex`，将切片正文转换为向量，来源元数据保存在文档库。需要 Python 3.12，运行 `uv sync` 安装依赖；依赖首次下载需要联网并占用磁盘。默认模型快照路径配置在 `src/rag/config/indexing.py`，CPU 本地推理，不发送文档或问题到外部服务，也不产生模型 API 费用。缓存不存在时需先从可信来源获取相同模型快照，或通过 `--model-path` 覆盖为其他快照目录。程序从缓存目录读取模型名和版本，并对快照文件计算 SHA-256 摘要；目录名本身不能证明模型来源，首次获取时仍应确认下载来源可信。

```bash
uv run python -m src.rag.cli.indexing build data/index-pages-4-5 docs/pdf/HowToLiveBetter.pdf \
  --start 4 --end 5 --chunk-size 128 --chunk-overlap 32
```

构建结束会列出片段 ID 和物理页码。重新加载索引时，可先查看所有文档的覆盖情况，包括没有任何可提取文本的文档：

```bash
uv run python -m src.rag.cli.indexing overview data/index-pages-4-5
```

选择一个片段 ID，重新读取完整正文和来源（无需重新加载 PDF）：

```bash
uv run python -m src.rag.cli.indexing read data/index-pages-4-5 片段ID
```

可在索引路径后传入多个 PDF；页码范围应用于每个文件。索引默认切片大小为 384、重叠为 64，单位是 LlamaIndex tokenizer 的 token；该经验值不保证适用于所有模型和文本。输出目录必须不存在，不能追加或覆盖：重复添加、内容或路径变更时，使用新目录并传入希望保留的全部文档进行受控重建。`manifest.json` 记录文档路径、路径标识、内容 SHA-256、页面范围、空文本页及文本覆盖状态、模型名称、版本和快照内容摘要、维度、切片配置及片段数。空文本页表明文本覆盖不完整，不代表空白页或解析正确；回查会再次提示。路径移动后旧索引仍记录旧位置，需重新构建；旧索引不会自动更新。切片超出嵌入模型长度时会拒绝构建，请减小切片大小。模型长度预检使用已加载的 SentenceTransformer 的 tokenizer 和长度上限，依赖当前锁定版本的 `HuggingFaceEmbedding` 内部模型接口，升级时需重新验证。失败分别报告向量化成功、失败和未处理的数量及索引发布状态；切片尚未生成时未处理数量为未知。临时构建不会作为完整索引发布。原子发布目前支持 macOS 和提供 `renameat2` 的 Linux；其他平台或文件系统不支持该操作时会报错。模型或切片参数更改时必须使用独立目录重建；当前阶段不提供查询或追加，供后续检索模块使用的配置比较函数会拒绝不兼容配置。

测试与类型检查：

```bash
uv run python -m unittest discover -s tests -v
uv run --with mypy python -m mypy --check-untyped-defs src tests
```

索引构建成功不代表语义检索质量达标。第二份 PDF、不同内容版本和路径变化的人工验收仍需后续检查；本地模型、索引和固定问题的检索观察见[检索实验记录](docs/experiments/retrieval.md)。

## 检索

检索使用已保存索引中的向量和 Node，不会重新解析 PDF。默认使用 `src/rag/config/indexing.py` 配置的 embedding 模型路径；它必须与构建索引时使用的快照一致。需要换模型目录时，可用 `--model-path` 或环境变量 `RAG_EMBEDDING_MODEL_PATH` 覆盖，并确保与索引配置相同。程序会比较模型名、revision、快照文件摘要、向量维度和切片配置，配置不兼容会拒绝查询。

```bash
uv run python -m src.rag.cli.retrieval search data/index-pages-4-5 \
  "如何改善生活质量？" \
  --top-k 5
```

默认结果按重排分数排序，同时显示 dense 原始排名/相似度分数、rerank 分数、片段 ID、完整正文、文件名、来源路径和 PDF 物理页码。两类分数不能互相比较，也不是答案正确率。若使用 `--dense-only`，结果才按向量相似度排序；dense 分数越高表示向量越相似，只能在同一索引和查询配置下比较。索引中缺少正文、文件名或合法物理页码的节点会被排除。源文件已移动或删除时，保存的正文仍会显示，但来源状态会标记为无法完成当前文件核对。

检索默认使用 `src/rag/config/retrieval.py` 中配置的本地 CrossEncoder `BAAI/bge-reranker-v2-m3`，执行 dense top 20 → rerank top 5。`--candidate-k` 是粗召回数量，默认 20；`--top-k` 是最终交付数量，默认 5。程序使用 `local_files_only=True`，不会联网下载，也不会把问题或文档发送到外部服务。配置路径不存在时会明确报错；可用 `--rerank-model-path` 临时覆盖配置，或加 `--dense-only` 运行 dense 基线：

首次使用需先安装模型到 Hugging Face 缓存（需要联网并占用约 2 GB），例如运行 `uv run hf download BAAI/bge-reranker-v2-m3`。默认配置指向本机当前缓存快照；其他机器或自定义安装目录可在运行命令前设置 `RAG_RERANK_MODEL_PATH=/path/to/model`，也可用 `--rerank-model-path` 覆盖。项目只在本地加载模型，不会自动下载。

```bash
uv run python -m src.rag.cli.retrieval search data/index-full-384 \
  "加班费、年休假该怎么算？" \
  --candidate-k 20 --top-k 5
```

输出会同时显示 dense 原始排名和分数、rerank 新排名和分数。两类分数来自不同模型，不能直接比较；重排只能调整已经进入前 20 的候选，不能找回 dense 阶段完全漏掉的片段。使用 `--dense-only` 可在相同问题上运行 dense 基线。

检索验收至少应包含三个文档内有明确依据的问题和一个文档内没有明确答案的问题。对每条结果打开对应 PDF，核对文件、物理页码和完整原文；无结果或来源缺失都要记录，不能仅凭相似度或重排分数判定回答正确。两阶段检索的实验协议和未完成的人工比较见[重排实验记录](docs/experiments/reranking.md)。基于来源的回答尚未实现。
