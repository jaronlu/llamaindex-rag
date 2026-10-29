"""运行日志默认配置。"""

from pathlib import Path

DEFAULT_LOG_FILE = Path('data/logs/rag.log')
DEFAULT_LOG_LEVEL = 'INFO'
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
