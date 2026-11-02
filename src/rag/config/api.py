"""局域网 HTTP 接口的默认配置。

这些值只影响 ``api`` 和 ``cli.serve``；检索与回答的算法参数仍由各自配置模块提供，
避免服务层另造一套默认值。
"""

# 默认只监听本机；手机访问必须显式改为 0.0.0.0，由启动命令承担风险提示。
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000

# 设置该环境变量后，除健康检查外的请求都需要携带 Bearer Token。
API_TOKEN_ENV = "RAG_API_TOKEN"

# 限制单次请求规模，避免超长问题或过大的候选数量占用模型资源。
# 上限只用于拒绝明显异常的请求，不是性能调优参数，因此不随命令行参数变化。
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 20
MAX_CANDIDATE_K = 100
