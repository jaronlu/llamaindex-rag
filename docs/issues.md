# 代码审查记录

## 2026-10-30 重排（reranking）改动（已校准）

- **审查范围**：`master` 工作区相对 `HEAD`（4e55e98）的未提交改动，共 11 个代码/文档文件：`README.md`、`docs/experiments/reranking.md`、`docs/prd/PRD.md`、`docs/superpowers/plans/2026-10-30-reranking.md`、`src/rag/cli/indexing.py`、`src/rag/cli/retrieval.py`、`src/rag/config/indexing.py`、`src/rag/config/retrieval.py`、`src/rag/retrieval/__init__.py`、`src/rag/retrieval/search.py`、`tests/test_retrieval.py`。
- **校准说明**：初版报告基于较早的工作区快照；此后工作区新增了「`--model-path` 改为可选并默认取本机 embedding 快照」的改动（`src/rag/config/indexing.py`、`src/rag/cli/indexing.py`、README、测试），本版按当前工作区重核了行号、结论和严重度，并新增 1 条发现。本文件自身不计入审查范围。
- **复核方式**：逐条对照源码/文档/plan 原文；用 `uv run python` 实测 numpy 的 `float()` 行为；检查两台默认模型路径在本机的存在性；核对 `uv.lock` 与包元数据。
- **核对结论**：共 11 条。10 条确认；原「多标签 CrossEncoder 崩溃」由推断升级为实测确认；1 条为校准期间新发现（默认 embedding 路径）。全部未修复。
- **严重度口径**：高 = 默认流程在新环境会失败或结果不可复现；中 = 违反项目约定、可观测的性能或文档错误；低 = 契约/可维护性问题，当前路径不易触发。

### 发现清单（按严重程度排序）

#### 1. 重排默认开启且模型路径写死本机快照，偏离 plan 的「可选」设计 — 高

- **位置**：`src/rag/config/retrieval.py:5-7`、`src/rag/cli/retrieval.py:34-46`
- **问题**：`DEFAULT_RERANK_MODEL_PATH` 硬编码了本机缓存中带固定 revision 的快照（`.../snapshots/953dc6f...`），`--rerank-model-path` 直接以它为默认值，且不传 `--dense-only` 就默认走重排。
- **依据**：`docs/superpowers/plans/2026-10-30-reranking.md` 的接口设计是「`--rerank-model-path` 为可选；传入时才重排，否则保持 dense 基线」，并未要求默认路径。实现把「可选开关」变成了「默认开启 + 本机默认路径」。该快照目录在开发机上存在，所以本机不会暴露问题。
- **影响**：新克隆仓库、只下载了嵌入模型的机器，运行 README 标准 `search` 时在 dense 检索前就报「重排模型不存在或不是目录」，命令直接失败；行为依赖开发者个人缓存状态，不可复现。
- **建议**：回到 plan 语义——不传 `--rerank-model-path` 即 dense 基线，传入才重排；若坚持默认开启，则默认值必须是可获取的模型标识（不含 revision 的模型族目录）并在 README 给出一条可复现的获取步骤。

#### 2. 默认 embedding 路径同样写死本机快照 — 高（校准期间新发现）

- **位置**：`src/rag/config/indexing.py:11-14`、`src/rag/cli/indexing.py:57`
- **问题**：原本的 `DEFAULT_MODEL`/`DEFAULT_REVISION` 是仓库级常量，本次改为 `Path.home()` 拼接的本机快照绝对路径，并让 `build`/`search` 的 `--model-path` 默认取它。
- **影响**：与发现 1 同类——新环境执行 README 的 `build`、`search` 示例时会在默认路径上失败；README 第 69 行已把「默认模型快照路径配置在 config」写成使用说明，进一步放大了这个隐含假设。另外 `DEFAULT_MODEL`/`DEFAULT_REVISION` 被删除，若外部仍引用会直接 `ImportError`（仓库内已无引用）。
- **建议**：保留模型名 + revision 作为配置常量，运行时在 Hugging Face 缓存中解析到实际快照；或要求显式 `--model-path`，README 示例写全路径。

#### 3. 重排实现未按 PRD 放入 `reranking/` 目录，plan 与 PRD 也不一致 — 中

- **位置**：`src/rag/retrieval/search.py:197`（`_load_cross_encoder`、`rerank`、`retrieve_and_rerank` 均在 `retrieval/search.py`）
- **问题**：`docs/prd/PRD.md:267` 规划了独立 `src/rag/reranking/`，但 plan 的 Task 1 直接把重排写进 `retrieval/search.py`，且未同步修改 PRD。
- **影响**：违反 AGENTS.md「新增或修改模块前对照 PRD 项目结构规划确定目录，完成后检查一致性；调整结构须说明原因并同步更新 PRD」。重排与检索耦合，无法作为独立模块替换/复用；文档与代码长期不一致。
- **建议**：迁移到 `src/rag/reranking/`；若确认不迁移，在 PRD 中删除该目录规划并写明原因。

#### 4. 20 个候选各触发一次整文件 SHA-256，其中 15 条随后被丢弃 — 中

- **位置**：`src/rag/retrieval/search.py:129`（`_source_status` 内的哈希在 106-112 行）
- **问题**：`retrieve` 对每个候选节点单独执行来源文件的完整 SHA-256，没有按 `source_path` 去重，也未推迟到重排之后。manifest 只读取一次，重复开销完全来自哈希。
- **影响**：单个 100 MB PDF 产生 20 个候选时，每次查询最多重复读取/哈希约 2 GB，最终只交付 5 条；改动前 `top_k=5` 把开销限制在 5 次。这是本次把候选数从 5 提到 20 带来的新增成本。
- **建议**：按 `source_path` 缓存核验结果（检索范围内同一文件只算一次），或只对重排后的最终结果核验来源；两种做法都要保留现有的「来源已变化/缺失」测试。

#### 5. README 首个检索示例默认走 rerank，紧随其后的输出说明仍只描述 dense — 中

- **位置**：`README.md:107-112`（rerank 默认说明在 114 行）
- **问题**：首个示例只有 `--top-k 5`，既没 `--dense-only` 也没 `--rerank-model-path`，按当前实现会走重排；紧接着第 112 行仍称输出只有相似度分数、且「分数越高表示向量越相似」。
- **影响**：照抄示例的用户在无默认快照时报错；有快照时输出含「原始排名/新排名/rerank 分数」，与 112 行描述矛盾，rerank 分数也不满足「越高越相似」的解释。
- **建议**：让首个示例与实现语义一致（发现 1 修复后天然一致），或在输出说明中按 dense/rerank 两种模式分别描述。

#### 6. 库函数签名写死 20/5，未复用 config 常量 — 中低

- **位置**：`src/rag/retrieval/search.py:263`（`retrieve_and_rerank` 的 `candidate_k=20, top_k=5`）、同文件 `rerank` 的 `top_k=5`
- **问题**：`src/rag/config/retrieval.py` 的常量只有 CLI 在用，库函数仍写字面量；CLI 帮助文本里的「默认 20」「默认 5」也是硬编码字符串。
- **依据**：这次 embedding 侧已经改为 CLI 与库共享 `DEFAULT_EMBEDDING_MODEL_PATH` 的模式（见 `tests/test_retrieval.py:204` 的新用例），重排侧没有对齐。
- **影响**：修改 `DEFAULT_CANDIDATE_K` 后 CLI 与库调用默认值不一致，帮助文本也可能与实际默认值脱节，违反 AGENTS.md「集中配置常量和默认值」。
- **建议**：库函数默认值引用 config 常量，帮助文本用 f-string 插值；或明确库层不提供默认值、由调用方必传。

#### 7. `float(raw_score)` 不支持多标签 CrossEncoder 输出 — 中低（已实测确认）

- **位置**：`src/rag/retrieval/search.py:245`
- **问题**：假定 `model.predict` 返回标量。多标签模型返回形状 `(n, k)` 的数组时，`float(np.array([...]))` 抛 `TypeError`。
- **依据**：`uv run python` 实测：`TypeError: only 0-dimensional arrays can be converted to Python scalars`（与 numpy 版本相关的措辞，本质都是 `TypeError`）。模块对外声明的输入错误是 `ValueError`。
- **影响**：`--rerank-model-path` 指向 `num_labels=2/3` 的本地 cross-encoder 时，查询以不透明异常中止，CLI 只打印「检索失败: only 0-dimensional arrays...」，破坏公开 API 的异常契约。
- **建议**：加载后校验 `num_labels == 1`（或显式指定 `activation_fn`/`num_labels`），并在 `predict` 结果形状不符时抛出带说明的 `ValueError`。

#### 8. 候选为空时 `rerank` 跳过模型校验直接返回 `[]` — 中低

- **位置**：`src/rag/retrieval/search.py:233-234`
- **问题**：`if not candidates: return []` 位于模型路径与可用性校验之前。
- **依据**：plan 的 Review Focus 明确要求「A missing reranker model path fails before model loading」，当前实现只在非空候选时满足。
- **影响**：直接调用导出的 `rerank('q', [], Path('/missing'))` 返回 `[]` 而不报错，与 PRD 6.5「模型缺失或不可用必须明确失败，不能静默降级」的意图不一致。CLI 编排路径因 `retrieve_and_rerank` 先校验重排路径，暂未触发。
- **建议**：把模型路径校验移到空候选短路之前，或在 docstring 中明确「空候选不加载模型」。

#### 9. 重排成功路径测试全部替换 `_load_cross_encoder` — 中低

- **位置**：`tests/test_retrieval.py:161`（另见 176-178 行编排测试整体 mock 掉 `rerank`）
- **问题**：真实的 `CrossEncoder` 构造、`local_files_only`/`trust_remote_code` 等 kwargs 以及默认模型路径从未被任何测试执行；「本地模型路径错误」用例只覆盖 `is_dir()` 分支。
- **影响**：sentence-transformers 升级后移除/重命名这些 kwargs，或默认路径写错时，`tests/test_retrieval.py` 的 15 个测试仍然全绿，而 CLI 运行时失败。
- **建议**：增加一个用极小本地 cross-encoder 夹具（或 patch `sentence_transformers.CrossEncoder` 断言构造参数）的测试。

#### 10. 实验记录声称覆盖「候选不足」，实际无用例 — 中低

- **位置**：`docs/experiments/reranking.md:29`
- **问题**：记录称自动化测试覆盖「候选不足」，但现有 rerank 用例是 3 个候选配 `top_k=2`（并非候选不足），编排测试固定返回 20 个候选且 mock 了 `rerank`；`test_validates_top_k_and_question` 的 `top_k=99` 只覆盖 dense 路径。
- **影响**：违反 AGENTS.md「报告检查范围，不得声称未运行的检查已通过」；plan 的 Review Focus 把「Fewer than 20 candidates」列为验收点，这里会误导后续验收。
- **建议**：补充「dense 返回少于 `candidate_k`」与「候选数小于最终 `top_k`」两组用例，或修正记录措辞。

#### 11. `sentence_transformers` 直接 import 但未在 `pyproject.toml` 声明 — 低

- **位置**：`src/rag/retrieval/search.py:209`
- **问题**：该包目前只是 `llama-index-embeddings-huggingface` 的传递依赖（`uv.lock` 中 `sentence-transformers 6.2.0`，上游声明 `>=2.6.1`），项目清单未显式声明。
- **影响**：上游调整传递依赖后 `uv sync` 仍成功，但重排会走到「缺少 sentence-transformers」的 `ValueError`；功能依赖无法从项目清单复现。项目已把上游锁在 `>=0.6,<0.7`，短期内风险有限。
- **建议**：在 `pyproject.toml` 中显式声明 `sentence-transformers` 并限定版本范围。

### 其他观察（不计入发现）

- `src/rag/cli/retrieval.py:42` 用 `assert args.rerank_model_path is not None` 做运行时校验，`python -O` 下会被移除；由于该参数已有默认值，这行本身也可删除。
- `tests/test_retrieval.py:200` 的 `if __name__ == "__main__": unittest.main()` 位于 `RetrievalCliTests` 定义之前，直接运行该文件会漏跑 CLI 测试。属既有问题，不在本次改动范围内。

### 建议修复顺序

1. 默认值可复现性：重排放回「可选」语义或提供可解析的默认模型；embedding 默认路径同法处理（发现 1、2）。
2. README 示例与输出说明对齐（发现 5）。
3. 按 PRD 迁移 `reranking/` 或同步 PRD 说明（发现 3）。
4. 来源核验按文件去重或后移（发现 4）。
5. 默认值集中、异常契约、空候选校验、依赖声明（发现 6、7、8、11）。
6. 补测试与修正实验记录（发现 9、10）。
