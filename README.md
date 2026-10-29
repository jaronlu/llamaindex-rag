# LlamaIndex RAG

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

本阶段使用本地 `BAAI/bge-small-zh-v1.5` 和 LlamaIndex `HuggingFaceEmbedding`、`VectorStoreIndex`，将切片正文转换为向量，来源元数据保存在文档库。需要 Python 3.12，运行 `uv sync` 安装依赖；依赖首次下载需要联网并占用磁盘。以下示例使用本机已缓存的模型快照，CPU 本地推理，不发送文档或问题到外部服务，也不产生模型 API 费用。缓存不存在时需先从可信来源获取相同模型快照；替换 `--model-path` 为 Hugging Face 缓存的快照目录。程序从缓存目录读取模型名和版本，并对快照文件计算 SHA-256 摘要；目录名本身不能证明模型来源，首次获取时仍应确认下载来源可信。

```bash
uv run python -m src.rag.cli.indexing build data/index-pages-4-5 docs/pdf/HowToLiveBetter.pdf \
  --start 4 --end 5 --chunk-size 128 --chunk-overlap 32 \
  --model-path "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
```

构建结束会列出片段 ID 和物理页码。重新加载索引时，可先查看所有文档的覆盖情况，包括没有任何可提取文本的文档：

```bash
uv run python -m src.rag.cli.indexing overview data/index-pages-4-5
```

选择一个片段 ID，重新读取完整正文和来源（无需重新加载 PDF）：

```bash
uv run python -m src.rag.cli.indexing read data/index-pages-4-5 片段ID
```

可在索引路径后传入多个 PDF；页码范围应用于每个文件。输出目录必须不存在，不能追加或覆盖：重复添加、内容或路径变更时，使用新目录并传入希望保留的全部文档进行受控重建。`manifest.json` 记录文档路径、路径标识、内容 SHA-256、页面范围、空文本页及文本覆盖状态、模型名称、版本和快照内容摘要、维度、切片配置及片段数。空文本页表明文本覆盖不完整，不代表空白页或解析正确；回查会再次提示。路径移动后旧索引仍记录旧位置，需重新构建；旧索引不会自动更新。切片超出嵌入模型长度时会拒绝构建，请减小切片大小。模型长度预检使用已加载的 SentenceTransformer 的 tokenizer 和长度上限，依赖当前锁定版本的 `HuggingFaceEmbedding` 内部模型接口，升级时需重新验证。失败分别报告向量化成功、失败和未处理的数量及索引发布状态；切片尚未生成时未处理数量为未知。临时构建不会作为完整索引发布。原子发布目前支持 macOS 和提供 `renameat2` 的 Linux；其他平台或文件系统不支持该操作时会报错。模型或切片参数更改时必须使用独立目录重建；当前阶段不提供查询或追加，供后续检索模块使用的配置比较函数会拒绝不兼容配置。

测试与类型检查：

```bash
uv run python -m unittest discover -s tests -v
uv run --with mypy python -m mypy --check-untyped-defs src tests
```

索引构建成功不代表语义检索质量达标。全量文档、第二份 PDF、不同内容版本和路径变化的人工验收以及检索质量比较仍需后续检查。
