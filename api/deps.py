from fastapi import HTTPException, Request


def get_session(request: Request):
    driver = request.app.state.neo4j_driver

    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j is not configured")

    with driver.session() as session:
        yield session


def get_weaviate(request: Request):
    client = request.app.state.weaviate

    if client is None:
        raise HTTPException(status_code=503, detail="Weaviate is not configured")

    return client


def get_nlp(request: Request):
    nlp = request.app.state.nlp

    if nlp is None:
        raise HTTPException(status_code=503, detail="NLP model is not configured")

    return nlp


def get_generator(request: Request):
    return request.app.state.generator


def get_embedder(request: Request):
    return request.app.state.embedder