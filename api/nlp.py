"""spaCy-backed entity extraction.

Used by the /extract path operation. The spaCy pipeline is constructed
once in `main.lifespan` and resolved via `Depends(get_nlp)` — do not
load the pipeline inside this module's functions.
"""
from .models import Entity


def extract_entities(text: str, nlp) -> list[Entity]:
    """Run spaCy NER on `text` and return entities ordered by `start`.

    Inputs:
        text — input string.
        nlp — loaded spaCy pipeline (passed in; do not load here).
    Returns:
        list[Entity] ordered by `start` ascending (the Evaluation
        Methodology requires monotonic non-decreasing `start`).
    """
    # TODO: call `doc = nlp(text)`, build Entity instances from
    #       `doc.ents`, and sort the result by `start` before returning.
    doc = nlp(text)
    entities = [Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char) for ent in doc.ents]
    return sorted(entities, key=lambda e: e.start)  

