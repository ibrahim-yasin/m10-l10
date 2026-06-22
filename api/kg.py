"""Wraps the vendored W9B deterministic NL→Cypher mapper.

The mapper itself is vendored under `w9b_mapper/` and must not be
modified. This module provides the thin function the path operation
calls.
"""
from .w9b_mapper.mapper import map_question
from .w9b_mapper.errors import UnsupportedQueryError  # re-export


def wrap_kg_query(question: str):
    """Map a natural-language question to (cypher, params).

    Inputs:
        question — natural-language question (non-empty, <= 500 chars).
    Returns:
        Tuple (cypher: str, params: dict) suitable for `session.run`.
    Raises:
        UnsupportedQueryError — if the question does not match any
        supported pattern. The path operation converts this to 422.
    """
    # TODO: call `map_question(question)` and return its (cypher, params).
    
    return map_question(question)   
