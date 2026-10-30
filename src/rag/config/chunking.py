"""所有切片入口共用的 token 默认值。

独立切片命令使用 512 token 作为通用演示默认值；直接构建向量索引使用 384，
为当前 embedding 模型留出 token 预算。两者用途不同，但集中定义以免被误认为重复配置。
"""

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
# 索引默认值更小，为 embedding 模型留出 token 预算；实际片段仍逐一校验上限。
DEFAULT_INDEX_CHUNK_SIZE = 384

# 结构边界：行首的项目符号或编号条目视为新块起点。
# PDF 抽取的正文通常只有单换行，LlamaIndex 默认的空行段落分隔不生效；没有这条规则时
# 切片会退化成纯句子切分，边界可能落在引文或 URL 中间，产生以残缺来源开头的片段。
STRUCTURE_BLOCK_START_REGEX = r"^[ \t]*(?:[•‣·▪●○◦]|[-*]\s|\d+[.、](?!\d))"
