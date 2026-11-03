"""基于来源回答的生成服务配置。

默认服务是火山方舟的 OpenAI 兼容接口，属外部服务：问题和检索片段会离开本机，
按服务商的计费规则产生费用。密钥只能从环境变量读取，本项目不保存、不打印密钥。
换服务或模型时修改这里，或在命令行用对应参数临时覆盖。
"""

DEFAULT_ANSWER_MODEL = "glm-5.3-flash"
DEFAULT_ANSWER_API_BASE = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_ANSWER_API_KEY_ENV = "ARK_WORK_FLOW_API_KEY"

# 回答需要稳定复现，温度调低；输出上限用于约束单次调用的费用和延迟。
# 当前模型会先输出推理内容再给答案，推理也计入输出上限，1024 容易把答案截断。
DEFAULT_ANSWER_TEMPERATURE = 0.1
DEFAULT_ANSWER_MAX_TOKENS = 4096
DEFAULT_ANSWER_TIMEOUT_SECONDS = 60.0

# 生成请求是否读取系统或环境变量里的代理。macOS 的系统代理会被 httpx 读取；
# 代理配置错误时表现为只有 "Connection error." 的连接失败，而直连往往正常。
# 保持默认行为（跟随系统和环境变量），需要绕过时用 --no-llm-proxy 显式关闭。
DEFAULT_ANSWER_TRUST_ENV = True

# 相关性门槛：只按排名取前 K，会把“你好”这类问题也塞进 5 条无关片段，生成模型
# 随后把它们总结成答案。分数来自 quickstart 索引的实测：文档内问题 dense ≥ 0.66、
# rerank ≥ 0.53，问候和书外问题 dense ≤ 0.42、rerank ≤ 0.20。重排分与 dense 分的
# 量纲不同，不能共用一个值；换模型或索引后必须重新校准。
DEFAULT_MIN_DENSE_SCORE = 0.5
DEFAULT_MIN_RERANK_SCORE = 0.3
