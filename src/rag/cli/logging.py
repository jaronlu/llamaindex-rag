"""命令行运行日志的配置与生命周期。"""

import contextvars
import functools
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ParamSpec, TypeVar

from ..config.logging import DEFAULT_LOG_FILE, DEFAULT_LOG_LEVEL, LOG_BACKUP_COUNT, LOG_MAX_BYTES

P = ParamSpec("P")
T = TypeVar("T")
_context: contextvars.ContextVar[tuple[str, str, int] | None] = contextvars.ContextVar('rag_log_context', default=None)
_lock = threading.RLock()
_handlers: dict[str, tuple[logging.Handler, int]] = {}
_active = 0
_original_level = 0


def bind_log_context(function: Callable[P, T]) -> Callable[P, T]:
    """提交线程任务时复制当前上下文；每个并发任务须单独调用。"""
    context = contextvars.copy_context()

    @functools.wraps(function)
    def bound(*args: P.args, **kwargs: P.kwargs) -> T:
        return context.run(function, *args, **kwargs)

    return bound


class RunFilter(logging.Filter):
    """只接收当前运行对应文件和级别的记录。"""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def filter(self, record: logging.LogRecord) -> bool:
        context = _context.get()
        if context is None or context[0] != self.path or record.levelno < context[2]:
            return False
        record.run_id = context[1]
        return True


class ReportingFileHandler(RotatingFileHandler):
    """写入失败时提示用户，不改变业务操作的结果。"""

    def handleError(self, record: logging.LogRecord) -> None:
        """每个 handler 只提示一次，避免磁盘故障刷屏。"""
        if not getattr(self, '_reported', False):
            self._reported = True
            print('警告：日志写入失败，后续诊断信息可能缺失。', file=sys.stderr)


def logged_command(function: Callable[P, int]) -> Callable[P, int]:
    """为一次命令配置文件日志，结束后关闭 handler。"""
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        global _active, _original_level
        logger = logging.getLogger('src.rag')
        raw_path = os.environ.get('RAG_LOG_FILE', str(DEFAULT_LOG_FILE))
        try:
            path: Path | None = Path(raw_path).expanduser().absolute()
        except (OSError, RuntimeError, ValueError) as exc:
            print(f'警告：无法解析日志文件 {raw_path}: {exc}', file=sys.stderr)
            path = None
        path_key = str(path) if path is not None else ''
        level_name = os.environ.get('RAG_LOG_LEVEL', DEFAULT_LOG_LEVEL).upper()
        levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
        if level_name not in levels:
            print(f'警告：无效日志级别 {level_name}，使用 {DEFAULT_LOG_LEVEL}。', file=sys.stderr)
            level_name = DEFAULT_LOG_LEVEL
        acquired = False
        with _lock:
            if _active == 0:
                _original_level = logger.level
                logger.setLevel(logging.DEBUG)
            _active += 1
            if path is None:
                pass
            elif path_key in _handlers:
                handler, count = _handlers[path_key]
                _handlers[path_key] = handler, count + 1
                acquired = True
            else:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handler = ReportingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
                    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s run_id=%(run_id)s %(name)s %(message)s'))
                    handler.addFilter(RunFilter(path_key))
                    logger.addHandler(handler)
                    _handlers[path_key] = handler, 1
                    acquired = True
                except OSError as exc:
                    print(f'警告：无法创建日志文件 {path}: {exc}', file=sys.stderr)
        token = _context.set((path_key, uuid.uuid4().hex, levels[level_name]))
        start = time.monotonic()
        try:
            logger.info('command.start command=%s', function.__module__)
            result = function(*args, **kwargs)
            logger.info('command.complete command=%s exit_code=%s duration_seconds=%.3f', function.__module__, result, time.monotonic() - start)
            return result
        except BaseException:
            logger.exception('command.failed command=%s duration_seconds=%.3f', function.__module__, time.monotonic() - start)
            raise
        finally:
            _context.reset(token)
            with _lock:
                if acquired:
                    handler, count = _handlers[path_key]
                    if count > 1:
                        _handlers[path_key] = handler, count - 1
                    else:
                        del _handlers[path_key]
                        logger.removeHandler(handler)
                        try:
                            handler.close()
                        except OSError as exc:
                            print(f'警告：关闭日志文件失败: {exc}', file=sys.stderr)
                _active -= 1
                if _active == 0:
                    logger.setLevel(_original_level)
    return wrapper
