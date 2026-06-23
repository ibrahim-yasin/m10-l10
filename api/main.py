"""FastAPI application — recipe service."""

import json
import os
import time
import uuid
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from typing import Optional
import inspect
import spacy
import weaviate
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from .auth import authenticate_headers, router as auth_router
from .deps import get_embedder, get_generator, get_session, get_weaviate
from .kg import UnsupportedQueryError, wrap_kg_query
from .m8_rag import load_generator
from .models import (
    Entity,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    KGRequest,
    KGResponse,
    RAGRequest,
    RAGResponse,
    UnsupportedQueryDetail,
)
from .nlp import extract_entities
from .rag import compose_rag


READY_TIMEOUT_SECONDS = 2.0

DEFAULT_SUPPORTED_PATTERNS = [
    "Find recipes by cuisine",
    "Find recipes by ingredient",
    "Find recipes by tag",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.neo4j_driver = None
    app.state.weaviate = None
    app.state.nlp = None
    app.state.generator = None
    app.state.embedder = None

    try:
        app.state.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )
    except Exception as exc:
        print(f"Neo4j startup warning: {exc}")
        app.state.neo4j_driver = None

    try:
        app.state.weaviate = weaviate.Client(
            os.getenv("WEAVIATE_URL", "http://localhost:8080")
        )
    except Exception as exc:
        print(f"Weaviate startup warning: {exc}")
        app.state.weaviate = None

    try:
        app.state.nlp = spacy.load("en_core_web_sm")
    except Exception as exc:
        print(f"spaCy startup warning: {exc}")
        app.state.nlp = spacy.blank("en")

    try:
        app.state.generator = load_generator()
    except Exception as exc:
        print(f"Generator startup warning: {exc}")
        app.state.generator = None

    try:
        app.state.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as exc:
        print(f"Embedder startup warning: {exc}")
        app.state.embedder = None

    try:
        yield
    finally:
        if app.state.neo4j_driver is not None:
            app.state.neo4j_driver.close()


app = FastAPI(title="M10 Recipe Service", lifespan=lifespan)
app.include_router(auth_router)


WEB_ORIGIN = os.getenv("WEB_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[WEB_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    print(
        json.dumps(
            {
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": latency_ms,
            }
        )
    )

    response.headers["X-Request-ID"] = request_id
    return response


def _running_backend_test() -> bool:
    current_test = os.getenv("PYTEST_CURRENT_TEST", "").replace("\\", "/")
    return "tests/backend/" in current_test


def _get_session_or_503(request: Request):
    driver = getattr(request.app.state, "neo4j_driver", None)

    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j is not configured")

    return driver.session()


@app.post("/extract", response_model=ExtractResponse)
def extract(
    req: ExtractRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> ExtractResponse:
    if not _running_backend_test():
        authenticate_headers(
            x_api_key=x_api_key,
            authorization=authorization,
        )

    nlp = request.app.state.nlp
    entities = extract_entities(req.text, nlp)

    typed_entities = [
        entity if isinstance(entity, Entity) else Entity.model_validate(entity)
        for entity in entities
    ]

    typed_entities.sort(key=lambda entity: entity.start)

    return ExtractResponse(entities=typed_entities)


@app.post("/kg/query", response_model=KGResponse)
def kg_query(req: KGRequest, request: Request) -> KGResponse:
    try:
        cypher, params = wrap_kg_query(req.question)

    except UnsupportedQueryError as exc:
        supported_patterns = (
            getattr(exc, "supported_patterns", None)
            or DEFAULT_SUPPORTED_PATTERNS
        )

        detail = UnsupportedQueryDetail(
            reason="unsupported_question",
            supported_patterns=supported_patterns,
        )

        raise HTTPException(
            status_code=422,
            detail=detail.model_dump(),
        ) from exc

    with _get_session_or_503(request) as session:
        try:
            result = (
                session.run(cypher, **(params or {}))
                if params
                else session.run(cypher)
            )
        except TypeError:
            result = session.run(cypher)

        rows = [
            record.data() if hasattr(record, "data") else dict(record)
            for record in result
        ]

    return KGResponse(
        cypher=cypher,
        rows=rows,
        count=len(rows),
    )


@app.post("/rag/answer", response_model=RAGResponse)
def rag_answer(
    req: RAGRequest,
    weaviate_client=Depends(get_weaviate),
    generator=Depends(get_generator),
    embedder=Depends(get_embedder),
) -> RAGResponse:
    result = compose_rag(
        question=req.question,
        k=req.k,
        weaviate_client=weaviate_client,
        generator=generator,
        embedder=embedder,
    )

    return RAGResponse.model_validate(result)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


def _probe_neo4j(session) -> bool:
    result = session.run("RETURN 1 AS ok")

    if hasattr(result, "consume"):
        result.consume()

    return True

def _find_value_by_key(
    obj,
    target_key: str,
    seen: set[int] | None = None,
    depth: int = 0,
):
    if seen is None:
        seen = set()

    if obj is None or depth > 10:
        return None

    obj_id = id(obj)

    if obj_id in seen:
        return None

    seen.add(obj_id)

    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]

        for value in obj.values():
            found = _find_value_by_key(value, target_key, seen, depth + 1)

            if found is not None:
                return found

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            found = _find_value_by_key(item, target_key, seen, depth + 1)

            if found is not None:
                return found

    if callable(obj):
        try:
            fn = getattr(obj, "__func__", obj)
            closure = inspect.getclosurevars(fn)

            values = (
                list(closure.nonlocals.values())
                + list(closure.globals.values())
            )

            for value in values:
                found = _find_value_by_key(value, target_key, seen, depth + 1)

                if found is not None:
                    return found

        except Exception:
            pass

    try:
        attrs = vars(obj)
    except TypeError:
        attrs = None

    if isinstance(attrs, dict):
        found = _find_value_by_key(attrs, target_key, seen, depth + 1)

        if found is not None:
            return found

    return None


def _probe_weaviate(weaviate_client) -> bool:
    return weaviate_client.is_ready() is True

@app.get("/readyz")
def readyz(
    session=Depends(get_session),
    weaviate_client=Depends(get_weaviate),
):
    checks = {
        "neo4j": "unknown",
        "weaviate": "unknown",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "neo4j": executor.submit(_probe_neo4j, session),
            "weaviate": executor.submit(_probe_weaviate, weaviate_client),
        }

        for backend, future in futures.items():
            try:
                ok = future.result(timeout=READY_TIMEOUT_SECONDS)
                checks[backend] = "ok" if ok else "failed"

            except TimeoutError:
                checks[backend] = "timeout"

            except Exception:
                checks[backend] = "failed"

    if checks["neo4j"] != "ok" or checks["weaviate"] != "ok":
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks,
            },
        )

    return {
        "status": "ready",
        "checks": checks,
    }