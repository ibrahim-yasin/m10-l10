"""FastAPI application — recipe service."""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager

import spacy
import weaviate
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from .deps import get_embedder, get_generator, get_nlp, get_session, get_weaviate
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
from .rag import answer_question


READY_TIMEOUT_SECONDS = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.neo4j_driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )

    app.state.weaviate = weaviate.Client(os.environ["WEAVIATE_URL"])
    app.state.nlp = spacy.load("en_core_web_sm")
    app.state.generator = load_generator()
    app.state.embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        yield
    finally:
        app.state.neo4j_driver.close()


app = FastAPI(title="M10 Recipe Service", lifespan=lifespan)


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


@app.post("/extract", response_model=ExtractResponse)
def extract(
    req: ExtractRequest,
    nlp=Depends(get_nlp),
) -> ExtractResponse:
    entities = extract_entities(req.text, nlp)

    typed_entities = [
        entity if isinstance(entity, Entity) else Entity.model_validate(entity)
        for entity in entities
    ]

    typed_entities.sort(key=lambda entity: entity.start)

    return ExtractResponse(entities=typed_entities)


@app.post("/kg/query", response_model=KGResponse)
def kg_query(
    req: KGRequest,
    session=Depends(get_session),
) -> KGResponse:
    try:
        cypher, params = wrap_kg_query(req.question)

    except UnsupportedQueryError as exc:
        detail = UnsupportedQueryDetail(
            reason="unsupported_question",
            supported_patterns=getattr(exc, "supported_patterns", []),
        )

        raise HTTPException(
            status_code=422,
            detail=detail.model_dump(),
        ) from exc

    result = session.run(cypher, params or {})
    rows = [record.data() for record in result]

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
    result = answer_question(
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
    result.consume()
    return True


def _probe_weaviate(weaviate_client) -> bool:
    return bool(weaviate_client.is_ready())


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