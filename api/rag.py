"""RAG composer — retrieve → assemble → generate → cite → grounding check."""

import re
from typing import Tuple


PROMPT_TEMPLATE = """\
You are answering a recipe question. Use ONLY the numbered sources below.
Cite each claim with the source number in square brackets, e.g. [1].
If the sources do not contain the answer, say: I cannot answer this from the available sources.

Sources:
{sources}

Question: {question}
Answer:"""

SENTINEL = "I cannot answer this from the available sources"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _to_vector(embedding):
    if hasattr(embedding, "tolist"):
        return embedding.tolist()

    return list(embedding)


def _chunk_score(chunk: dict) -> float:
    distance = chunk.get("_additional", {}).get("distance", 1.0)

    try:
        score = 1.0 - float(distance)
    except (TypeError, ValueError):
        score = 0.0

    return max(0.0, min(1.0, score))


def assemble_prompt(question: str, chunks: list[dict]) -> Tuple[str, dict[int, dict]]:
    numbered = {
        i + 1: chunk
        for i, chunk in enumerate(chunks)
    }

    sources = "\n".join(
        f"[{i}] {chunk.get('text', '')}"
        for i, chunk in numbered.items()
    )

    prompt = PROMPT_TEMPLATE.format(
        sources=sources,
        question=question,
    )

    return prompt, numbered


def extract_citations(answer: str, numbered: dict[int, dict]) -> list[dict]:
    citations = []
    seen = set()

    for match in CITATION_PATTERN.finditer(answer):
        source_number = int(match.group(1))

        if source_number in seen:
            continue

        if source_number not in numbered:
            continue

        seen.add(source_number)

        chunk = numbered[source_number]

        citations.append(
            {
                "chunk_id": int(chunk["chunk_id"]),
                "score": _chunk_score(chunk),
            }
        )

    return citations


def _extract_generated_text(generator_output) -> str:
    if isinstance(generator_output, str):
        return generator_output

    if isinstance(generator_output, list) and generator_output:
        first = generator_output[0]

        if isinstance(first, dict):
            return str(
                first.get("generated_text")
                or first.get("summary_text")
                or first.get("text")
                or ""
            )

        return str(first)

    if isinstance(generator_output, dict):
        return str(
            generator_output.get("generated_text")
            or generator_output.get("summary_text")
            or generator_output.get("text")
            or ""
        )

    return str(generator_output or "")


def _retrieve_chunks(question: str, embedder, weaviate_client, k: int) -> list[dict]:
    vector = _to_vector(embedder.encode(question))

    response = (
        weaviate_client.query
        .get("Chunk", ["text", "chunk_id"])
        .with_near_vector({"vector": vector})
        .with_additional(["distance"])
        .with_limit(k)
        .do()
    )

    return response.get("data", {}).get("Get", {}).get("Chunk", [])


def compose_rag(
    question: str,
    embedder,
    weaviate_client,
    generator,
    k: int = 4,
) -> dict:
    retrieved = _retrieve_chunks(
        question=question,
        embedder=embedder,
        weaviate_client=weaviate_client,
        k=k,
    )

    if not retrieved:
        return {
            "answer": SENTINEL,
            "citations": [],
            "confidence": 0.0,
        }

    prompt, numbered = assemble_prompt(question, retrieved)

    generator_output = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False,
    )

    answer = _extract_generated_text(generator_output).strip()

    citations = extract_citations(answer, numbered)

    if not citations:
        return {
            "answer": SENTINEL,
            "citations": [],
            "confidence": 0.0,
        }

    confidence = sum(c["score"] for c in citations) / len(citations)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }