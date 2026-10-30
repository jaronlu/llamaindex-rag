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
