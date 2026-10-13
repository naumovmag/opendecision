"""Локальный HTTP API с одной моделью и ограничением входных данных."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .backends.base import ContextLengthError
from .config import Settings
from .engine import DecisionEngine, create_engine
from .schemas import BatchRequest, DecisionRequest, DecisionResponse

logger = logging.getLogger("opendecision")


def create_app(engine: DecisionEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            app.state.engine = create_engine(Settings.load(
                os.getenv("OPENDECISION_CONFIG", "configs/inference.yaml")))
            if app.state.engine.settings.warmup and hasattr(app.state.engine.backend, "warmup"):
                seconds = app.state.engine.backend.warmup()
                logger.warning('{"event":"warmup","seconds":%.1f}', seconds)
        else:
            app.state.engine = engine
        yield
        if engine is None:
            app.state.engine.backend.close()

    app = FastAPI(title="OpenDecision", lifespan=lifespan)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1048576:
                return JSONResponse(status_code=413, content={"detail": "Лимит тела: 1 MiB"})
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    def service(request: Request) -> DecisionEngine:
        return request.app.state.engine

    @app.get("/health")
    def health(svc: Annotated[DecisionEngine, Depends(service)]):
        return {"status": "ok", "model": svc.settings.public_name}

    @app.get("/v1/models")
    def models(svc: Annotated[DecisionEngine, Depends(service)]):
        return {"models": [svc.backend.metadata()]}

    async def decide_all(requests, svc):
        # Async: ожидание не занимает поток, все одновременные запросы попадают к батчеру.
        outcomes = await asyncio.gather(*(asyncio.wrap_future(f) for f in svc.submit(requests)),
                                        return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, ContextLengthError):
                raise HTTPException(422, str(outcome)) from outcome
            if isinstance(outcome, Exception):
                logger.error("Ошибка inference", exc_info=outcome)
                raise HTTPException(503, "Модель временно недоступна") from outcome
        for result in outcomes:
            logger.info('{"event":"decision","latency_ms":%.3f}', result.latency_ms)
        return outcomes

    @app.post("/v1/decisions", response_model=DecisionResponse)
    async def decide(payload: DecisionRequest, svc: Annotated[DecisionEngine, Depends(service)]):
        return (await decide_all([payload], svc))[0]

    @app.post("/v1/decisions/batch", response_model=list[DecisionResponse])
    async def batch(payload: BatchRequest, svc: Annotated[DecisionEngine, Depends(service)]):
        return await decide_all(payload.requests, svc)

    return app


app = create_app()
