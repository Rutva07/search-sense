"""FastAPI service. Inference reads history; only /observe updates it."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis
from .engine import SearchEngine
from .history import RedisHistory


class CompletionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    prefix: str = Field(min_length=1, max_length=512)
    timestamp: int | None = Field(default=None, ge=0)
    k: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True


class ObservationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=512)
    timestamp: int | None = Field(default=None, ge=0)


def create_app(artifacts=None, redis_url=None, namespace=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = SearchEngine(Path(artifacts or os.environ.get("SEARCHSENSE_ARTIFACTS", "artifacts")),
                                       RedisHistory(redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                                                    namespace or os.environ.get("SEARCHSENSE_NAMESPACE", "searchsense")))
        yield
        app.state.engine.history.client.close()

    app = FastAPI(title="SearchSense", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        try:
            app.state.engine.history.client.ping()
            return {"status": "ok", "model_version": app.state.engine.version}
        except redis.RedisError:
            raise HTTPException(503, "History store unavailable")

    @app.post("/complete")
    def complete(request: CompletionRequest):
        try:
            return app.state.engine.complete(request.user_id, request.prefix,
                                             request.timestamp if request.timestamp is not None else int(time.time()),
                                             request.k, request.use_cache)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except redis.RedisError:
            raise HTTPException(503, "History store unavailable")

    @app.post("/observe")
    def observe(request: ObservationRequest):
        try:
            app.state.engine.observe(request.user_id, request.query,
                                      request.timestamp if request.timestamp is not None else int(time.time()))
            return {"status": "recorded"}
        except ValueError as e:
            raise HTTPException(422, str(e))
        except redis.RedisError:
            raise HTTPException(503, "History store unavailable")
    return app


app = create_app()
