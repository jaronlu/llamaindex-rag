# 代码审查记录

## 2026-10-31 全链路评审问题处理（来源：外部 review 报告）

- **范围**：`src/rag/`、`tests/`、`docs/`、`README.md`；外部报告文件核对后已删除，结论保留在本节。
- **处理原则**：逐条对照当前源码复核；属实项修复并补测试，设计取舍项保留并记录理由，未复现项注明。
- **结论**：无高严重度问题；3 个中等问题和多数低级别问题属实并已处理。

| 编号 | 报告问题 | 复核结果 | 处理 |
| --- | --- | --- | --- |
| M1 | 查询兼容性检查中维度/切片三项取自 manifest 自身再与 manifest 比较，恒真；README 夸大 | 属实 | `check_compatibility` 只比较 `expected` 中显式给出的字段，`retrieve` 只传模型身份并注明理由；README 检索小节同步 |
| M2 | README 写“四个阶段命令行入口”，实际已有 5 个 | 属实 | 改为五个阶段，并补“回答”事件 |
| M3 | 发现 8 的“空候选也会加载模型”与当前实现相反 | 属实 | 在发现 8 补 2026-10-31 校准说明，保留原记录作为历史 |
| L1 | `retrieve` docstring 只声明 `ValueError` | 属实 | 补充模型目录缺失时的 `FileNotFoundError` |
| L2 | 原文中的 `[数字]` 会被误判为引用编号 | 属实（设计边界） | 在 `generate.py` 和 `docs/knowledge/07-answer-issues.md` 记录误报边界；只提示、不阻塞 |
| L3 | `split.py` 页码冗余三元表达式；`page_label` 与 `format_pages` 重复 | 属实 | 删除不可达分支；页码格式化统一到 `retrieval.search.format_pages` |
| L4 | darwin 分支未捕获 `renamex_np` 缺失 | 属实（当前机器不触发） | 与 Linux 分支一致，捕获 `AttributeError` 并抛 `OSError` |
| L5 | 重排模型无快照内容摘要校验 | 属实，已知缺口 | 维持现状，`docs/knowledge/06-rerank-issues.md` 已有记录 |
| L6 | 评测每题重复加载模型和索引 | 属实，设计取舍 | 维持现状，理由见 `evaluate_questions` docstring |
| L7 | 评测编排、CLI 输出、截断 SDK 形态、非文本 content 的测试缺口 | 部分属实 | 新增评测编排、SDK 形态截断、非文本 content 测试；CLI 评测输出仍无直接测试，列为剩余缺口 |
| L8 | 回答生成失败时不展示已检索证据 | 属实 | 失败分支打印已取得的证据 |
| L9 | `.mimosa/`、`.v2c/` 未被忽略 | 当前已不存在 | 工作区 `.gitignore` 已加入两项（已暂存），本次未改动；8.2MB 测试 PDF 入库为已知取舍 |

- **未处理/未验证**：未实跑 rerank 与回答链路（回答会产生外部费用）；PRD §5 的第二份 PDF 扩展验收仍未完成，与本次评审问题无关，继续保留在实验记录的未验证清单中。
- **其他观察**：项目目前只有单元测试和 mypy，没有独立 lint 配置；属观察项，不是违规。本次已在本机对现有索引实跑 dense 检索，结果仍能回到 PDF 第 4 页，确认兼容性检查调整未影响正常查询。

## 2026-10-30 重排（reranking）改动（已校准）

- **审查范围**：`master` 工作区相对 `HEAD`（4e55e98）的未提交改动，共 11 个代码/文档文件：`README.md`、`docs/experiments/reranking.md`、`docs/prd/PRD.md`、`docs/superpowers/plans/2026-10-30-reranking.md`、`src/rag/cli/indexing.py`、`src/rag/cli/retrieval.py`、`src/rag/config/indexing.py`、`src/rag/config/retrieval.py`、`src/rag/retrieval/__init__.py`、`src/rag/retrieval/search.py`、`tests/test_retrieval.py`。
- **校准说明**：初版报告基于较早的工作区快照；此后工作区新增了「`--model-path` 改为可选并默认取本机 embedding 快照」的改动（`src/rag/config/indexing.py`、`src/rag/cli/indexing.py`、README、测试），本版按当前工作区重核了行号、结论和严重度，并新增 1 条发现。本文件自身不计入审查范围。
- **复核方式**：逐条对照源码/文档/plan 原文；用 `uv run python` 实测 numpy 的 `float()` 行为；检查两台默认模型路径在本机的存在性；核对 `uv.lock` 与包元数据。
- **本轮复核结论**：原报告的 11 条都描述了当时真实存在的情况；其中 1、2 项的可移植性风险已通过环境变量覆盖和模型准备说明处理，同时保留用户选择的本机默认模型路径。其余问题已在本轮或之前修复。无误报；原报告的事实不应被改写成当时不存在。
- **当前状态**：11 条均已处理；具体处理方式见各条状态。CrossEncoder 尚未使用真实 PDF 完成质量验收。
- **严重度口径**：高 = 默认流程在新环境会失败或结果不可复现；中 = 违反项目约定、可观测的性能或文档错误；低 = 契约/可维护性问题，当前路径不易触发。

### 发现清单（按严重程度排序）

#### 1. 重排默认开启且模型路径写死本机快照，偏离 plan 的「可选」设计 — 高

- **状态：已处理（保留默认重排，补足跨机器配置和模型准备说明）**。用户明确选择默认配置更方便，因此不恢复“必须显式传参才启用”的旧方案。新增 `RAG_RERANK_MODEL_PATH` 覆盖，并在 README 说明如何下载和指定模型；缺失时仍明确报错，不自动联网下载。

- **位置**：`src/rag/config/retrieval.py:5-7`、`src/rag/cli/retrieval.py:34-46`
- **问题**：`DEFAULT_RERANK_MODEL_PATH` 硬编码了本机缓存中带固定 revision 的快照（`.../snapshots/953dc6f...`），`--rerank-model-path` 直接以它为默认值，且不传 `--dense-only` 就默认走重排。
- **依据**：`docs/superpowers/plans/2026-10-30-reranking.md` 的接口设计是「`--rerank-model-path` 为可选；传入时才重排，否则保持 dense 基线」，并未要求默认路径。实现把「可选开关」变成了「默认开启 + 本机默认路径」。该快照目录在开发机上存在，所以本机不会暴露问题。
- **影响**：新克隆仓库、只下载了嵌入模型的机器，运行 README 标准 `search` 时在 dense 检索前就报「重排模型不存在或不是目录」，命令直接失败；行为依赖开发者个人缓存状态，不可复现。
- **建议**：回到 plan 语义——不传 `--rerank-model-path` 即 dense 基线，传入才重排；若坚持默认开启，则默认值必须是可获取的模型标识（不含 revision 的模型族目录）并在 README 给出一条可复现的获取步骤。

#### 2. 默认 embedding 路径同样写死本机快照 — 高（校准期间新发现）

- **状态：已处理（保留本机默认值，增加可配置路径与下载指引）**。新增 `RAG_EMBEDDING_MODEL_PATH` 覆盖，README 提供默认 revision 的下载命令。`DEFAULT_MODEL`/`DEFAULT_REVISION` 当时在仓库内无其他引用，不属于当前仓库内部导入错误。

- **位置**：`src/rag/config/indexing.py:11-14`、`src/rag/cli/indexing.py:57`
- **问题**：原本的 `DEFAULT_MODEL`/`DEFAULT_REVISION` 是仓库级常量，本次改为 `Path.home()` 拼接的本机快照绝对路径，并让 `build`/`search` 的 `--model-path` 默认取它。
- **影响**：与发现 1 同类——新环境执行 README 的 `build`、`search` 示例时会在默认路径上失败；README 第 69 行已把「默认模型快照路径配置在 config」写成使用说明，进一步放大了这个隐含假设。另外 `DEFAULT_MODEL`/`DEFAULT_REVISION` 被删除，若外部仍引用会直接 `ImportError`（仓库内已无引用）。
- **建议**：保留模型名 + revision 作为配置常量，运行时在 Hugging Face 缓存中解析到实际快照；或要求显式 `--model-path`，README 示例写全路径。

#### 3. 重排实现未按 PRD 放入 `reranking/` 目录，plan 与 PRD 也不一致 — 中

- **状态：已修复**。重排当前与 dense 查询共同编排，PRD 结构规划已调整为 `retrieval/` 同时承载检索、候选重排和来源过滤，并注明属于同一查询流程。

- **位置**：`src/rag/retrieval/search.py:197`（`_load_cross_encoder`、`rerank`、`retrieve_and_rerank` 均在 `retrieval/search.py`）
- **问题**：`docs/prd/PRD.md:267` 规划了独立 `src/rag/reranking/`，但 plan 的 Task 1 直接把重排写进 `retrieval/search.py`，且未同步修改 PRD。
- **影响**：违反 AGENTS.md「新增或修改模块前对照 PRD 项目结构规划确定目录，完成后检查一致性；调整结构须说明原因并同步更新 PRD」。重排与检索耦合，无法作为独立模块替换/复用；文档与代码长期不一致。
- **建议**：迁移到 `src/rag/reranking/`；若确认不迁移，在 PRD 中删除该目录规划并写明原因。

#### 4. 20 个候选各触发一次整文件 SHA-256，其中 15 条随后被丢弃 — 中

- **状态：已修复**。`retrieve()` 现在在单次查询内按 `source_path` 缓存来源核验结果；回归测试确认同一来源的多个候选只调用一次校验。

- **位置**：`src/rag/retrieval/search.py:129`（`_source_status` 内的哈希在 106-112 行）
- **问题**：`retrieve` 对每个候选节点单独执行来源文件的完整 SHA-256，没有按 `source_path` 去重，也未推迟到重排之后。manifest 只读取一次，重复开销完全来自哈希。
- **影响**：单个 100 MB PDF 产生 20 个候选时，每次查询最多重复读取/哈希约 2 GB，最终只交付 5 条；改动前 `top_k=5` 把开销限制在 5 次。这是本次把候选数从 5 提到 20 带来的新增成本。
- **建议**：按 `source_path` 缓存核验结果（检索范围内同一文件只算一次），或只对重排后的最终结果核验来源；两种做法都要保留现有的「来源已变化/缺失」测试。

#### 5. README 首个检索示例默认走 rerank，紧随其后的输出说明仍只描述 dense — 中

- **状态：已修复**。README 现在说明默认结果按重排分数排序，并分别解释两种分数；dense 相似度排序说明仅适用于 `--dense-only`。

- **位置**：`README.md:107-112`（rerank 默认说明在 114 行）
- **问题**：首个示例只有 `--top-k 5`，既没 `--dense-only` 也没 `--rerank-model-path`，按当前实现会走重排；紧接着第 112 行仍称输出只有相似度分数、且「分数越高表示向量越相似」。
- **影响**：照抄示例的用户在无默认快照时报错；有快照时输出含「原始排名/新排名/rerank 分数」，与 112 行描述矛盾，rerank 分数也不满足「越高越相似」的解释。
- **建议**：让首个示例与实现语义一致（发现 1 修复后天然一致），或在输出说明中按 dense/rerank 两种模式分别描述。

#### 6. 库函数签名写死 20/5，未复用 config 常量 — 中低

- **状态：已修复**。`rerank()`、`retrieve_and_rerank()` 默认参数现引用集中配置常量，CLI help 也从相同常量生成默认值。

- **位置**：`src/rag/retrieval/search.py:263`（`retrieve_and_rerank` 的 `candidate_k=20, top_k=5`）、同文件 `rerank` 的 `top_k=5`
- **问题**：`src/rag/config/retrieval.py` 的常量只有 CLI 在用，库函数仍写字面量；CLI 帮助文本里的「默认 20」「默认 5」也是硬编码字符串。
- **依据**：这次 embedding 侧已经改为 CLI 与库共享 `DEFAULT_EMBEDDING_MODEL_PATH` 的模式（见 `tests/test_retrieval.py:204` 的新用例），重排侧没有对齐。
- **影响**：修改 `DEFAULT_CANDIDATE_K` 后 CLI 与库调用默认值不一致，帮助文本也可能与实际默认值脱节，违反 AGENTS.md「集中配置常量和默认值」。
- **建议**：库函数默认值引用 config 常量，帮助文本用 f-string 插值；或明确库层不提供默认值、由调用方必传。

#### 7. `float(raw_score)` 不支持多标签 CrossEncoder 输出 — 中低（已实测确认）

- **状态：已修复**。非标量输出现在转换为说明清楚的 `ValueError`；回归测试覆盖多标签分数组合，避免向调用方泄漏底层 `TypeError`。

- **位置**：`src/rag/retrieval/search.py:245`
- **问题**：假定 `model.predict` 返回标量。多标签模型返回形状 `(n, k)` 的数组时，`float(np.array([...]))` 抛 `TypeError`。
- **依据**：`uv run python` 实测：`TypeError: only 0-dimensional arrays can be converted to Python scalars`（与 numpy 版本相关的措辞，本质都是 `TypeError`）。模块对外声明的输入错误是 `ValueError`。
- **影响**：`--rerank-model-path` 指向 `num_labels=2/3` 的本地 cross-encoder 时，查询以不透明异常中止，CLI 只打印「检索失败: only 0-dimensional arrays...」，破坏公开 API 的异常契约。
- **建议**：加载后校验 `num_labels == 1`（或显式指定 `activation_fn`/`num_labels`），并在 `predict` 结果形状不符时抛出带说明的 `ValueError`。

#### 8. 候选为空时 `rerank` 跳过模型校验直接返回 `[]` — 中低

- **状态：已修复（2026-10-31 校准）**。空候选会先检查模型目录，缺失时明确失败。审查当时的实现还会加载模型；后续为“返回空列表不值得加载数 GB 的 CrossEncoder”改为只校验路径、不加载模型（`src/rag/retrieval/search.py` 的 `rerank`），并由 `test_rerank_skips_model_load_when_no_candidates` 断言加载器未被调用。上面初版记录写作“并加载模型”，描述的是当时状态，特此校准。

- **位置**：`src/rag/retrieval/search.py:233-234`
- **问题**：`if not candidates: return []` 位于模型路径与可用性校验之前。
- **依据**：plan 的 Review Focus 明确要求「A missing reranker model path fails before model loading」，当前实现只在非空候选时满足。
- **影响**：直接调用导出的 `rerank('q', [], Path('/missing'))` 返回 `[]` 而不报错，与 PRD 6.5「模型缺失或不可用必须明确失败，不能静默降级」的意图不一致。CLI 编排路径因 `retrieve_and_rerank` 先校验重排路径，暂未触发。
- **建议**：把模型路径校验移到空候选短路之前，或在 docstring 中明确「空候选不加载模型」。

#### 9. 重排成功路径测试全部替换 `_load_cross_encoder` — 中低

- **状态：已修复**。新增加载器测试，patch 实际 `sentence_transformers.CrossEncoder` 构造入口，并断言本地路径、CPU、`local_files_only=True` 和 `trust_remote_code=False` 参数。

- **位置**：`tests/test_retrieval.py:161`（另见 176-178 行编排测试整体 mock 掉 `rerank`）
- **问题**：真实的 `CrossEncoder` 构造、`local_files_only`/`trust_remote_code` 等 kwargs 以及默认模型路径从未被任何测试执行；「本地模型路径错误」用例只覆盖 `is_dir()` 分支。
- **影响**：sentence-transformers 升级后移除/重命名这些 kwargs，或默认路径写错时，`tests/test_retrieval.py` 的 15 个测试仍然全绿，而 CLI 运行时失败。
- **建议**：增加一个用极小本地 cross-encoder 夹具（或 patch `sentence_transformers.CrossEncoder` 断言构造参数）的测试。

#### 10. 实验记录声称覆盖「候选不足」，实际无用例 — 中低

- **状态：已修复**。新增编排测试验证 dense 实际候选少于请求数，以及重排候选少于最终 Top-K 时只返回实际候选；实验记录已列明具体覆盖项。

- **位置**：`docs/experiments/reranking.md:29`
- **问题**：记录称自动化测试覆盖「候选不足」，但现有 rerank 用例是 3 个候选配 `top_k=2`（并非候选不足），编排测试固定返回 20 个候选且 mock 了 `rerank`；`test_validates_top_k_and_question` 的 `top_k=99` 只覆盖 dense 路径。
- **影响**：违反 AGENTS.md「报告检查范围，不得声称未运行的检查已通过」；plan 的 Review Focus 把「Fewer than 20 candidates」列为验收点，这里会误导后续验收。
- **建议**：补充「dense 返回少于 `candidate_k`」与「候选数小于最终 `top_k`」两组用例，或修正记录措辞。

#### 11. `sentence_transformers` 直接 import 但未在 `pyproject.toml` 声明 — 低

- **状态：已修复**。`sentence-transformers` 已加入项目直接依赖，并更新 `uv.lock`。

- **位置**：`src/rag/retrieval/search.py:209`
- **问题**：该包目前只是 `llama-index-embeddings-huggingface` 的传递依赖（`uv.lock` 中 `sentence-transformers 6.2.0`，上游声明 `>=2.6.1`），项目清单未显式声明。
- **影响**：上游调整传递依赖后 `uv sync` 仍成功，但重排会走到「缺少 sentence-transformers」的 `ValueError`；功能依赖无法从项目清单复现。项目已把上游锁在 `>=0.6,<0.7`，短期内风险有限。
- **建议**：在 `pyproject.toml` 中显式声明 `sentence-transformers` 并限定版本范围。

### 其他观察（不计入发现）

- `src/rag/cli/retrieval.py:42` 用 `assert args.rerank_model_path is not None` 做运行时校验：**已修复**，参数已设置默认值，无需 assert。
- `tests/test_retrieval.py` 中 `unittest.main()` 原先位于 CLI 测试类之前，直接运行文件会漏跑该类：**已修复**，入口已移至文件末尾。

### 原报告建议顺序（现状）

1. 默认值可复现性：保留用户要求的默认模型路径，已增加可覆盖环境变量和带固定 revision 的下载说明（发现 1、2）。
2. README 示例与输出说明已对齐（发现 5）。
3. PRD 已同步目录职责（发现 3）。
4. 来源核验已按文件路径去重（发现 4）。
5. 默认值集中、非标量分数错误、空候选模型校验、直接依赖声明均已修复（发现 6、7、8、11）。
6. 模型加载参数与候选不足测试已补齐，实验记录已更新（发现 9、10）。
