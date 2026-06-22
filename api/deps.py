"""FastAPI dependency-injection helpers.

These functions resolve process-scoped resources (Neo4j driver, Weaviate
client, spaCy pipeline, flan-t5-base generator, sentence-transformers
embedder) that were constructed once in `main.lifespan` and stored on
`app.state`. Path operations declare these via `Depends()` so the
resource is fetched per request without being re-constructed.

The autograder pins the attribute names — `app.state.neo4j_driver`,
`app.state.weaviate_client`, `app.state.generator`, `app.state.nlp`,
`app.state.embedder` — so use those exact names in both `main.lifespan`
and the helpers below.
"""
from urllib import request

from fastapi import Request


async def get_session(request: Request):
    """Yield a short-lived Neo4j session from the process-scoped driver.

    Yield one session per request; the `with` block closes it on exit.
    The driver itself was constructed once in `main.lifespan` and stashed
    on `app.state` — pull it from there.
    """
    # TODO: pull the Neo4j driver off `request.app.state`, open a
    #       session inside a `with` block, and yield it.
    async with request.app.state.neo4j_driver.session() as session:
        yield session   
    


def get_weaviate(request: Request):
    """Return the process-scoped Weaviate client constructed in lifespan."""
    # TODO: return the Weaviate client stashed on `request.app.state`.
    return request.app.state.weaviate_client    

def get_generator(request: Request):
    """Return the process-scoped flan-t5-base generator constructed in lifespan."""
    # TODO: return the generator stashed on `request.app.state`.
    return request.app.state.generator


def get_nlp(request: Request):
    """Return the process-scoped spaCy pipeline constructed in lifespan."""
    # TODO: return the spaCy pipeline stashed on `request.app.state`.
    return request.app.state.nlp    

def get_embedder(request: Request):
    """Return the process-scoped sentence-transformers embedder.

    `/rag/answer` uses this to encode the query into the same vector
    space as the chunks seeded by `seed_weaviate.py`, so
    `with_near_vector` returns meaningful results against a
    `vectorizer=none` Weaviate class.
    """
    # TODO: return the embedder stashed on `request.app.state`.
    return request.app.state.embedder