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

当前仅安装文档加载阶段所需的 LlamaIndex 核心库和文件读取集成，尚未实现加载模块。依赖版本记录在 `uv.lock`。
