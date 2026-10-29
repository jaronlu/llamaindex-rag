import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag.cli.loading import main as load_main
from src.rag.cli.chunking import main as chunk_main


class LoggingTests(unittest.TestCase):
    def test_cli_records_correlated_stages_without_document_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.log'
            env = {'RAG_LOG_FILE': str(path)}
            with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(chunk_main(['docs/pdf/HowToLiveBetter.pdf', '--start', '2', '--end', '2']), 0)
            log = path.read_text(encoding='utf-8')
            self.assertIn('command.start', log)
            self.assertIn('loading.complete', log)
            self.assertIn('chunking.complete', log)
            self.assertIn('command.complete', log)
            ids = {line.split('run_id=')[1].split()[0] for line in log.splitlines() if 'run_id=' in line}
            self.assertEqual(len(ids), 1)
            self.assertNotIn(next(line for line in output.getvalue().splitlines() if len(line) > 80 and not line.startswith(('文件:', '---'))), log)

    def test_failure_has_traceback_and_repeated_calls_do_not_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.log'
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(path)}), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(load_main(['missing.pdf']), 1)
                self.assertEqual(load_main(['missing.pdf']), 1)
            log = path.read_text(encoding='utf-8')
            self.assertEqual(log.count('command.start'), 2)
            self.assertEqual(log.count('loading.failed'), 2)
            self.assertIn('Traceback', log)

    def test_unwritable_log_is_reported_without_changing_result(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'RAG_LOG_FILE': temp}), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(load_main(['missing.pdf']), 1)
            self.assertIn('日志', err.getvalue())
            self.assertIn('加载失败', err.getvalue())


class LoggingBoundaryTests(unittest.TestCase):
    def test_failed_page_records_location_and_stack(self):
        from pypdf import PdfWriter
        from pypdf._page import PageObject
        with tempfile.TemporaryDirectory() as temp:
            pdf = Path(temp) / 'page.pdf'
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(pdf)
            log = Path(temp) / 'run.log'
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(log)}), patch.object(PageObject, 'extract_text', side_effect=ValueError('bad page')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(load_main([str(pdf)]), 1)
            content = log.read_text()
            self.assertIn('loading.page_failed', content)
            self.assertIn('page=1', content)
            self.assertIn('bad page', content)
            self.assertIn('Traceback', content)

    def test_index_success_and_failure_record_publication_state(self):
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.schema import TextNode
        from src.rag.cli.logging import logged_command
        from src.rag.indexing.store import build_index
        class FailingEmbedding(MockEmbedding):
            def _get_text_embedding(self, text):
                raise ValueError('embedding failed')
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'run.log'
            @logged_command
            def run(argv=None):
                build_index([TextNode(text='private text')], MockEmbedding(embed_dim=8), Path(temp) / 'ok', {})
                try:
                    build_index([TextNode(text='private text')], FailingEmbedding(embed_dim=8), Path(temp) / 'bad', {})
                except ValueError:
                    pass
                return 0
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(log)}):
                self.assertEqual(run(), 0)
            content = log.read_text()
            self.assertIn('index.published', content)
            self.assertIn('published=False', content)
            self.assertIn('embedding failed', content)
            self.assertNotIn('private text', content)

    def test_log_write_failure_does_not_abort_command(self):
        from src.rag.cli.logging import logged_command
        @logged_command
        def run(argv=None):
            return 0
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(Path(temp) / 'run.log')}), patch('logging.handlers.RotatingFileHandler.doRollover', side_effect=OSError('disk full')), patch('src.rag.cli.logging.LOG_MAX_BYTES', 1), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(run(), 0)
            self.assertIn('日志写入失败', err.getvalue())

    def test_close_failure_preserves_result_and_logger_state(self):
        import logging
        from src.rag.cli.logging import ReportingFileHandler, logged_command
        logger = logging.getLogger('src.rag')
        original = (logger.level, logger.propagate, list(logger.handlers))
        @logged_command
        def run():
            return 0
        def close_then_fail(handler: ReportingFileHandler) -> None:
            super(ReportingFileHandler, handler).close()
            raise OSError('close failed')
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RAG_LOG_FILE': str(Path(temp) / 'log')}), patch.object(ReportingFileHandler, 'close', close_then_fail), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(run(), 0)
        self.assertIn('日志', err.getvalue())
        self.assertEqual((logger.level, logger.propagate, logger.handlers), original)

    def test_concurrent_commands_keep_events_in_their_own_run(self):
        import logging
        import threading
        from src.rag.cli.logging import logged_command
        barrier = threading.Barrier(2)
        @logged_command
        def run(name):
            logging.getLogger('src.rag.test').info('marker=%s', name)
            barrier.wait(timeout=5)
            logging.getLogger('src.rag.test').info('end_marker=%s', name)
            barrier.wait(timeout=5)
            return 0
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'log'
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(log)}):
                threads = [threading.Thread(target=run, args=(name,)) for name in ('alpha', 'beta')]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
            lines = log.read_text().splitlines()
            markers = [line for line in lines if 'marker=' in line]
            self.assertEqual(len(markers), 4)
            groups: dict[str, set[str]] = {}
            for line in markers:
                run_id = line.split('run_id=')[1].split()[0]
                name = line.rsplit('=', 1)[1]
                groups.setdefault(run_id, set()).add(name)
            self.assertEqual(len(groups), 2)
            self.assertEqual({frozenset(group) for group in groups.values()}, {frozenset({'alpha'}), frozenset({'beta'})})

    def test_published_index_cleanup_failure_is_reported_consistently(self):
        from llama_index.core.embeddings import MockEmbedding
        from src.rag.cli.indexing import main
        from src.rag.ingestion.pdf import LoadResult, PageResult
        from src.rag.config.ingestion import PageStatus
        from llama_index.core import Document
        from pydantic import PrivateAttr
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1]
        class LocalEmbedding(MockEmbedding):
            max_length: int = 512
            _model: object = PrivateAttr(default_factory=lambda: type('LocalModel', (), {'tokenizer': Tokenizer(), 'prompts': {}})())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / 'source.pdf'
            pdf.write_bytes(b'pdf')
            model = root / 'models--a--b' / 'snapshots' / 'v1'
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            doc = Document(text='text', metadata={'file_name': 'source.pdf', 'source_path': str(pdf), 'source_id': 'a', 'page_numbers': [1]})
            loaded = LoadResult(pdf, 1, [PageResult(1, PageStatus.SUCCESS, doc)])
            log = root / 'run.log'
            # 清理阶段抛错，同时保留临时目录上下文的实际清理行为。
            original = tempfile.TemporaryDirectory.__exit__
            def failing_exit(instance, *args):
                original(instance, *args)
                raise OSError('cleanup failed')
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(log)}), patch('src.rag.cli.indexing.load_pdf', return_value=loaded), patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding', return_value=LocalEmbedding(embed_dim=8)), patch.object(tempfile.TemporaryDirectory, '__exit__', failing_exit), contextlib.redirect_stderr(io.StringIO()) as err:
                result = main(['build', str(root / 'index'), str(pdf), '--model-path', str(model)])
            self.assertEqual(result, 1)
            self.assertTrue((root / 'index').is_dir())
            self.assertIn('索引已发布', err.getvalue())
            command_line = next(line for line in log.read_text().splitlines() if 'index.command_failed' in line)
            self.assertIn('published=True', command_line)

    def test_external_library_logs_still_reach_callers_during_command(self):
        import logging
        import threading
        from src.rag.cli.logging import logged_command
        ready, done = threading.Event(), threading.Event()
        @logged_command
        def run():
            ready.set()
            done.wait(timeout=5)
            return 0
        logger = logging.getLogger('src.rag')
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        old_level, old_propagate = logger.level, logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = True
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'RAG_LOG_FILE': str(Path(temp) / 'run.log')}):
                thread = threading.Thread(target=run)
                thread.start()
                self.assertTrue(ready.wait(timeout=5))
                logging.getLogger('src.rag.ingestion.pdf').warning('external library event')
                done.set()
                thread.join(timeout=5)
            self.assertIn('external library event', stream.getvalue())
        finally:
            done.set()
            root.removeHandler(handler)
            logger.setLevel(old_level)
            logger.propagate = old_propagate

    def test_invalid_home_in_log_path_does_not_block_command(self):
        with patch.dict(os.environ, {'RAG_LOG_FILE': '~rag_nonexistent_user_92341/run.log'}), contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
            result = load_main(['docs/pdf/HowToLiveBetter.pdf', '--start', '2', '--end', '2'])
        self.assertEqual(result, 0)
        self.assertIn('日志', err.getvalue())

    def test_explicit_context_copy_correlates_worker_thread(self):
        import logging
        import threading
        from src.rag.cli.logging import logged_command, bind_log_context
        @logged_command
        def run():
            logger = logging.getLogger('src.rag.worker')
            logger.info('parent event')
            def work() -> None:
                logger.warning('worker event')
            worker = threading.Thread(target=bind_log_context(work))
            worker.start()
            worker.join(timeout=5)
            return 0
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'run.log'
            with patch.dict(os.environ, {'RAG_LOG_FILE': str(log)}):
                self.assertEqual(run(), 0)
            lines = log.read_text().splitlines()
            parent = next(line for line in lines if 'parent event' in line)
            worker = next(line for line in lines if 'worker event' in line)
            self.assertEqual(parent.split('run_id=')[1].split()[0], worker.split('run_id=')[1].split()[0])
