"""FastAPI 应用：向局域网客户端提供健康检查、检索和回答接口。

服务端负责全部文档处理和外部模型调用；客户端只发送问题并展示结果。这里不实现
新的检索或生成逻辑，而是复用 ``retrieval.search`` 和 ``answering.generate``，
保证 HTTP 结果与命令行结果使用同一套来源核验和引用检查规则。
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..answering.generate import (
    Answer,
    answer_from_text,
    create_llm,
    generate_answer,
    iter_answer_deltas,
    resolve_api_key,
    select_relevant_evidence,
)
from ..config.answering import (
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_TEMPERATURE,
    DEFAULT_ANSWER_TIMEOUT_SECONDS,
    DEFAULT_MIN_DENSE_SCORE,
    DEFAULT_MIN_RERANK_SCORE,
)
from ..config.api import API_TOKEN_ENV, MAX_CANDIDATE_K, MAX_QUESTION_CHARS, MAX_TOP_K
from ..config.indexing import ManifestKey
from ..config.retrieval import DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_TOP_K
from ..provenance import read_manifest
from ..retrieval.search import RetrievalResult, retrieve, retrieve_and_rerank


logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """一次服务运行的固定配置。

    索引路径和模型路径由启动命令提供；生成模型参数与命令行共享同一组默认值。
    请求级参数（Top-K、是否重排）不在这里，而是随每次请求传入。
    """

    index: Path
    embed_model: Path
    rerank_model: Path
    answer_model: str
    answer_api_base: str
    answer_api_key_env: str
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE


class SearchRequest(BaseModel):
    """检索和回答请求；字段含义与命令行同名参数一致。

    ``dense_only`` 为真时跳过重排，此时 ``candidate_k`` 不生效。``stream`` 只对
    ``/ask`` 有效：为真时返回 Server-Sent Events，为假时返回一次性 JSON。
    """

    question: str
    top_k: int = DEFAULT_RETRIEVAL_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    dense_only: bool = False
    stream: bool = False


def _expected_token() -> str:
    """读取当前进程配置的访问令牌；空字符串表示不鉴权。

    每次请求都重新读取环境变量，便于测试和长期运行时调整令牌，
    不必重启服务即可生效。
    """
    return os.environ.get(API_TOKEN_ENV, "").strip()


def _require_token(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """校验可选的 Bearer Token。

    未配置令牌时不做限制，由启动命令提示仅限可信局域网；配置后使用固定时间比较，
    避免通过响应时间推断令牌内容。
    """
    expected = _expected_token()
    if not expected:
        return
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="缺少或错误的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate(request: SearchRequest) -> None:
    """校验请求规模，把可预期的输入问题转成明确的 400。

    Pydantic 负责字段类型；这里补充中文提示和业务上限，避免把超长问题送进模型。
    超限时直接拒绝，不截断后继续，否则客户端会以为服务完整处理了问题。
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"问题长度不能超过 {MAX_QUESTION_CHARS} 个字符",
        )
    if not 1 <= request.top_k <= MAX_TOP_K:
        raise HTTPException(status_code=400, detail=f"top_k 必须在 1–{MAX_TOP_K} 之间")
    if not 1 <= request.candidate_k <= MAX_CANDIDATE_K:
        raise HTTPException(
            status_code=400,
            detail=f"candidate_k 必须在 1–{MAX_CANDIDATE_K} 之间",
        )
    if not request.dense_only and request.top_k > request.candidate_k:
        raise HTTPException(status_code=400, detail="top_k 不能大于 candidate_k")


def _require_index(settings: ApiSettings) -> None:
    """确认索引目录已完整发布；缺失或损坏时返回 404 而不是 500。"""
    try:
        read_manifest(settings.index)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _retrieve(settings: ApiSettings, request: SearchRequest) -> list[RetrievalResult]:
    """按请求模式调用现有检索实现，并把服务端配置错误映射为 503。

    这里不吞掉异常：检索失败时不能返回空列表冒充“没有相关片段”，
    否则客户端无法区分“资料里没有”和“服务端模型不可用”。
    """
    question = request.question.strip()
    try:
        if request.dense_only:
            return retrieve(settings.index, settings.embed_model, question, top_k=request.top_k)
        return retrieve_and_rerank(
            settings.index,
            settings.embed_model,
            settings.rerank_model,
            question,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"模型目录不存在: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
    """把检索结果转换成客户端可展示的 JSON；页码和来源状态与命令行保持一致。"""
    return {
        "rank": result.rank,
        "score": result.score,
        "rerank_score": result.rerank_score,
        "original_rank": result.original_rank,
        "node_id": result.node_id,
        "text": result.text,
        "file_name": result.file_name,
        "source_path": result.source_path,
        "page_numbers": list(result.page_numbers),
        "page_label": result.page_label,
        "source_exists": result.source_exists,
        "provenance_complete": result.provenance_complete,
        "provenance_warning": result.provenance_warning,
        "source_status": result.source_status_label,
    }


def _serialize_answer(answer: Answer) -> dict[str, Any]:
    """把回答和证据转换成 JSON；引用检查结果由客户端原样展示。"""
    return {
        "question": answer.question,
        "answer": answer.answer,
        "citations": list(answer.citations),
        "invalid_citations": list(answer.invalid_citations),
        "llm_called": answer.llm_called,
        "evidence": [_serialize_result(item) for item in answer.evidence],
    }


def _serialize_results(results: Sequence[RetrievalResult]) -> list[dict[str, Any]]:
    """批量序列化，保持交付顺序不变。"""
    return [_serialize_result(result) for result in results]


def _sse(event: str, data: dict[str, Any]) -> str:
    """把一条事件编码成 Server-Sent Events 帧。

    ``json.dumps`` 会把文本里的换行转义掉，保证 ``data`` 始终是单行；否则一段
    带换行的回答会把客户端按空行切分事件的逻辑直接打乱。中文不转义，便于调试时
    直接读懂事件内容。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _answer_events(settings: ApiSettings, request: SearchRequest) -> Iterator[str]:
    """按检索→证据→生成→完成的顺序产出 SSE 事件。

    生成器只在流开始后被消费，因此这里的失败无法再改变 HTTP 状态码；检索或生成
    出错时发送 `error` 事件并结束，不继续输出半截结果。
    """
    # 先发状态，客户端立刻知道请求已被接收，不必等服务端加载模型。
    yield _sse("status", {"stage": "retrieving", "message": "正在检索证据"})
    try:
        results = _retrieve(settings, request)
    except HTTPException as exc:
        yield _sse("error", {"detail": exc.detail})
        return

    # 只按排名取前 K 会把无关片段交给模型；先按相关度过滤，全部低于门槛时
    # 下面会走“资料不足”分支，不调用生成模型。
    results = select_relevant_evidence(
        results,
        min_rerank_score=settings.min_rerank_score,
        min_dense_score=settings.min_dense_score,
    )

    # 证据先于生成结果发出，客户端可以先渲染来源，再等待回答文本。
    yield _sse("evidence", {"items": _serialize_results(results)})
    if not results:
        yield _sse("done", _serialize_answer(generate_answer(request.question, results)))
        return

    yield _sse("status", {"stage": "generating", "message": f"正在基于 {len(results)} 条证据生成回答"})
    try:
        api_key = resolve_api_key(settings.answer_api_key_env)
        llm = create_llm(
            model=settings.answer_model,
            api_base=settings.answer_api_base,
            api_key=api_key,
            temperature=DEFAULT_ANSWER_TEMPERATURE,
            max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
            timeout=DEFAULT_ANSWER_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        yield _sse("error", {"detail": str(exc)})
        return
    except Exception as exc:
        logger.exception("api.stream_llm_setup_failed model=%s", settings.answer_model)
        yield _sse("error", {"detail": f"创建生成模型失败: {exc}"})
        return

    parts: list[str] = []
    try:
        for delta in iter_answer_deltas(request.question, results, llm):
            parts.append(delta)
            yield _sse("delta", {"text": delta})
        answer = answer_from_text(request.question, results, "".join(parts))
    except Exception as exc:
        # 只记录问题长度和证据数量，不记录问题正文或片段内容。
        logger.exception(
            "api.stream_answer_failed model=%s question_length=%s evidence=%s",
            settings.answer_model,
            len(request.question),
            len(results),
        )
        yield _sse("error", {"detail": f"生成回答失败: {exc}"})
        return

    logger.info(
        "api.stream_answer model=%s question_length=%s evidence=%s citations=%s",
        settings.answer_model,
        len(request.question),
        len(results),
        len(answer.citations),
    )
    yield _sse("done", _serialize_answer(answer))


def _stream_response(settings: ApiSettings, request: SearchRequest) -> StreamingResponse:
    """创建 SSE 响应，并关闭可能缓存流内容的中间层。"""
    return StreamingResponse(
        _answer_events(settings, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(settings: ApiSettings) -> FastAPI:
    """创建带检索、回答和健康检查路由的应用。

    索引和模型路径固定在 ``settings`` 中，请求只能调整问题、Top-K 和是否重排；
    客户端不能借此读取任意本机文件或替换模型。
    """
    app = FastAPI(title="LlamaIndex RAG 局域网接口", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """把 FastAPI 默认的 422 统一成带中文说明的 400。

        客户端只关心“请求哪里不对”，逐字段的英文校验详情对手机端展示没有帮助，
        而且与业务校验的 400 混用两套语气容易让人误判。
        """
        return JSONResponse(status_code=400, content={"detail": "请求格式或字段类型不正确"})

    @app.get("/health")
    def health() -> dict[str, Any]:
        """返回索引和鉴权状态，供客户端判断服务是否可用。

        索引不可用时仍返回 200 和 ``status: unavailable``：这样客户端能区分
        “连不上服务”和“服务在但索引坏了”，并展示具体原因。
        """
        base: dict[str, Any] = {
            "auth_required": bool(_expected_token()),
            "index": str(settings.index.expanduser().resolve()),
            "rerank_model_available": settings.rerank_model.expanduser().is_dir(),
        }
        try:
            manifest = read_manifest(settings.index)
        except ValueError as exc:
            return {**base, "status": "unavailable", "error": str(exc)}
        return {
            **base,
            "status": "ok",
            "chunk_count": manifest.get(ManifestKey.CHUNK_COUNT),
            "model": manifest.get(ManifestKey.MODEL),
            "version": manifest.get(ManifestKey.VERSION),
            "dimension": manifest.get(ManifestKey.DIMENSION),
        }

    @app.post("/search", dependencies=[Depends(_require_token)])
    def search(request: SearchRequest) -> dict[str, Any]:
        """返回带来源的候选片段，不调用生成模型。"""
        _validate(request)
        _require_index(settings)
        results = _retrieve(settings, request)
        logger.info(
            "api.search mode=%s question_length=%s results=%s",
            "dense" if request.dense_only else "rerank",
            len(request.question),
            len(results),
        )
        return {
            "question": request.question.strip(),
            "mode": "dense" if request.dense_only else "dense+rerank",
            "top_k": request.top_k,
            "candidate_k": request.candidate_k,
            "results": _serialize_results(results),
        }

    # 同一路由可能返回 JSON 或 StreamingResponse，关闭自动响应模型推断。
    @app.post("/ask", dependencies=[Depends(_require_token)], response_model=None)
    def ask(request: SearchRequest) -> Response | dict[str, Any]:
        """检索证据并调用生成模型；``stream`` 决定返回 SSE 还是完整 JSON。

        没有证据时不请求外部服务，两种模式都直接给出资料不足提示。流式模式下
        鉴权、参数和索引检查仍在这里完成，保证开始发送前还能返回普通 4xx。
        """
        _validate(request)
        _require_index(settings)
        if request.stream:
            return _stream_response(settings, request)
        results = _retrieve(settings, request)
        results = select_relevant_evidence(
            results,
            min_rerank_score=settings.min_rerank_score,
            min_dense_score=settings.min_dense_score,
        )
        if not results:
            # 与命令行一致：没有可交付片段时直接返回资料不足，不要求配置密钥。
            return _serialize_answer(generate_answer(request.question, results))

        try:
            api_key = resolve_api_key(settings.answer_api_key_env)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            llm = create_llm(
                model=settings.answer_model,
                api_base=settings.answer_api_base,
                api_key=api_key,
                temperature=DEFAULT_ANSWER_TEMPERATURE,
                max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
                timeout=DEFAULT_ANSWER_TIMEOUT_SECONDS,
            )
            answer = generate_answer(request.question, results, llm)
        except Exception as exc:
            # 只记录问题长度和证据数量，不记录问题正文或片段内容。
            logger.exception(
                "api.answer_failed model=%s question_length=%s evidence=%s",
                settings.answer_model,
                len(request.question),
                len(results),
            )
            raise HTTPException(status_code=502, detail=f"生成回答失败: {exc}") from exc

        logger.info(
            "api.answer model=%s question_length=%s evidence=%s citations=%s",
            settings.answer_model,
            len(request.question),
            len(results),
            len(answer.citations),
        )
        return _serialize_answer(answer)

    return app
