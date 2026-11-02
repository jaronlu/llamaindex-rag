import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag.retrieval.evaluate import (
    EvaluationQuestion,
    answer_rank,
    evaluate_questions,
    load_questions,
)
from src.rag.provenance import format_pages
from src.rag.retrieval.search import RetrievalResult


def setUpModule():
    """把评估测试的日志写入临时文件，避免污染用户的运行日志。"""
    temp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(temp.cleanup)
    env = patch.dict(os.environ, {'RAG_LOG_FILE': str(Path(temp.name) / 'test.log'), 'RAG_LOG_LEVEL': 'INFO'})
    env.start()
    unittest.addModuleCleanup(env.stop)


def result(rank, pages, score=0.5):
    """构造一条只关心排名和页码的检索结果。"""
    return RetrievalResult(
        rank=rank, score=score, node_id=f"node-{rank}", text="正文", file_name="a.pdf",
        source_path="/a.pdf", page_numbers=tuple(pages), source_exists=True, provenance_complete=True,
    )


class QuestionSetTests(unittest.TestCase):
    def test_loads_questions_and_keeps_outside_topics_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "questions.json"
            path.write_text(json.dumps([
                {"id": "a", "question": "问题一？", "answer_pages": [291, 292]},
                {"id": "b", "question": "书外问题？", "answer_pages": []},
            ]), encoding="utf-8")
            questions = load_questions(path)
        self.assertEqual(questions[0], EvaluationQuestion("a", "问题一？", (291, 292)))
        self.assertEqual(questions[1].answer_pages, ())

    def test_rejects_duplicate_ids_and_invalid_pages(self):
        cases = [
            ([{"id": "a", "question": "一？", "answer_pages": [1]}, {"id": "a", "question": "二？", "answer_pages": [2]}], "重复"),
            ([{"id": "a", "question": "一？", "answer_pages": [0]}], "正整数"),
            ([{"id": "a", "question": "一？", "answer_pages": [True]}], "正整数"),
            ([{"id": "a", "question": "一？"}], "正整数"),
            ([{"question": "一？", "answer_pages": []}], "id"),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "questions.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_questions(path)

    def test_rejects_empty_or_unreadable_file(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "非空"):
                load_questions(empty)
            with self.assertRaisesRegex(ValueError, "无法读取"):
                load_questions(Path(temp) / "missing.json")


class AnswerRankTests(unittest.TestCase):
    def test_cross_page_chunk_counts_as_hit(self):
        results = [result(1, [290]), result(2, [291, 292]), result(3, [293])]
        self.assertEqual(answer_rank(results, [292]), 2)

    def test_missing_answer_returns_none_and_outside_topics_never_hit(self):
        self.assertIsNone(answer_rank([result(1, [10]), result(2, [11])], [291]))
        self.assertIsNone(answer_rank([result(1, [10])], []))

    def test_format_pages(self):
        self.assertEqual(format_pages(()), "无")
        self.assertEqual(format_pages((291,)), "291")
        self.assertEqual(format_pages((291, 292)), "291–292")


class EvaluateQuestionsTests(unittest.TestCase):
    def test_records_dense_and_rerank_ranks_through_production_functions(self):
        question = EvaluationQuestion("q", "问题？", (2,))
        candidates = [result(1, [1]), result(2, [2]), result(3, [3])]
        reranked = [result(1, [3]), result(2, [2]), result(3, [1])]

        with patch("src.rag.retrieval.evaluate.retrieve", return_value=candidates) as retrieve_mock, \
             patch("src.rag.retrieval.evaluate.rerank", return_value=reranked) as rerank_mock:
            evaluations = evaluate_questions(
                Path("/index"), Path("/model"), Path("/reranker"), [question],
                candidate_k=10, top_k=5,
            )

        retrieve_mock.assert_called_once_with(Path("/index"), Path("/model"), "问题？", top_k=10)
        rerank_mock.assert_called_once_with("问题？", candidates, Path("/reranker"), top_k=5)
        evaluation = evaluations[0]
        self.assertEqual(evaluation.dense_rank, 2)
        self.assertEqual(evaluation.rerank_rank, 2)
        self.assertTrue(evaluation.in_candidates)
        self.assertEqual(evaluation.dense_pages, ((1,), (2,), (3,)))

    def test_distinguishes_not_recalled_from_not_reranked(self):
        question = EvaluationQuestion("q", "问题？", (9,))
        candidates = [result(1, [1]), result(2, [2])]

        with patch("src.rag.retrieval.evaluate.retrieve", return_value=candidates), \
             patch("src.rag.retrieval.evaluate.rerank", return_value=candidates[:1]):
            evaluation = evaluate_questions(
                Path("/index"), Path("/model"), Path("/reranker"), [question], top_k=1,
            )[0]

        self.assertFalse(evaluation.in_candidates)
        self.assertIsNone(evaluation.dense_rank)
        self.assertIsNone(evaluation.rerank_rank)


if __name__ == "__main__":
    unittest.main()
