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
uv run python -m src.rag.cli docs/pdf/HowToLiveBetter.pdf --start 2 --end 3
```

省略页码参数可检查全文；替换路径即可加载其他 PDF。输出包含完整页面文本、文件路径、物理页码和成功、空文本、失败页统计。对照 PDF 检查中文和段落顺序；空文本页需要人工确认。失败页不影响其他页面处理，但命令返回非零状态。暂不支持 OCR，扫描件或图片中的文字可能无法提取。

```bash
uv run python -m unittest discover -s tests -v
```

本模块使用 LlamaIndex `BaseReader` 和 `Document`，通过 `pypdf` 逐页提取文本。这样可以选择页码范围并报告单页失败；`Document.metadata` 保留文件路径、文件标识及物理页码。文件标识由绝对路径生成，移动文件后标识会变化。
