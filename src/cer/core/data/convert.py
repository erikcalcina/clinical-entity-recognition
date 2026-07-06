"""Dataset format converters shared across the three approaches.

Two on-disk JSON shapes are used throughout the repo:

- **GLiNER training shape** — word-tokenised text with word-index spans::

      {"tokenized_text": ["A", "31-year-old", ...],
       "ner": [[1, 1, "Age"], [4, 5, "Disease disorder"], ...]}

- **Span/eval shape** — raw text with entity strings::

      {"text": "A 31-year-old man ...",
       "entities": [{"text": "31-year-old", "label": "Age", "start": 2, "end": 13}, ...]}

The GLiNER trainer consumes the training shape; the LLM/GLiNER evaluators consume
the span/eval shape (``entities``); the BERT approach consumes a BIO shape derived
here (``{"tokens": [...], "ner_tags": ["O", "B-Age", ...]}``).

The functions below convert between these without depending on any model library,
so they import cleanly with only the base dependencies installed.
"""

import re
from typing import Dict, List, Tuple


def detect_entity_key(records: List[Dict]) -> str:
    """Return the key holding the entity list in span/eval records.

    Different corpora in this project use either ``"entities"`` or ``"labels"``; both
    hold a list of ``{"text", "label"}`` dicts. Returns ``"entities"`` for empty input.
    """
    for record in records:
        if "entities" in record:
            return "entities"
        if "labels" in record:
            return "labels"
    return "entities"


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Whitespace-tokenise ``text`` and return ``(tokens, char_offsets)``."""
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def gliner_ner_to_bio(record: Dict) -> Dict:
    """Convert a GLiNER training record to a BIO record.

    Spans use inclusive word-level indices ``[start, end, label]``. On overlap the
    first span (sorted by start index) wins.
    """
    tokens = record["tokenized_text"]
    bio = ["O"] * len(tokens)
    for start, end, label in sorted(record.get("ner", []), key=lambda x: x[0]):
        if bio[start] != "O":
            continue
        bio[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            if bio[i] == "O":
                bio[i] = f"I-{label}"
    return {"tokens": tokens, "ner_tags": bio}


def spans_to_bio(record: Dict) -> Dict:
    """Convert a span/eval record (raw text + entity strings) to a BIO record.

    Entity strings are matched to whitespace tokens via character offsets. A token
    is covered when its character start falls inside the entity's character span,
    which correctly handles punctuation attached to the final token
    (e.g. ``"metastases."``). Repeated entity strings are matched left to right.
    """
    text = record["text"]
    tokens, offsets = tokenize_with_offsets(text)
    bio = ["O"] * len(tokens)

    def first_occurrence(entry: Dict) -> int:
        idx = text.find(entry["text"])
        return idx if idx != -1 else len(text)

    search_pos: Dict[str, int] = {}
    for entry in sorted(record.get("entities", record.get("labels", [])), key=first_occurrence):
        entity_text, label = entry["text"], entry["label"]
        char_start = text.find(entity_text, search_pos.get(entity_text, 0))
        if char_start == -1:
            continue
        char_end = char_start + len(entity_text)
        search_pos[entity_text] = char_end

        token_start = token_end = None
        for ti, (ts, _te) in enumerate(offsets):
            if char_start <= ts < char_end:
                if token_start is None:
                    token_start = ti
                token_end = ti
        if token_start is None:
            continue
        if bio[token_start] == "O":
            bio[token_start] = f"B-{label}"
            for i in range(token_start + 1, token_end + 1):
                if bio[i] == "O":
                    bio[i] = f"I-{label}"
    return {"tokens": tokens, "ner_tags": bio}


def bio_to_entities(tokens: List[str], bio_tags: List[str]) -> List[Dict]:
    """Reconstruct ``[{"text", "label"}, ...]`` spans from word-level BIO tags."""
    entities, cur_tokens, cur_label = [], [], None
    for token, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if cur_tokens:
                entities.append({"text": " ".join(cur_tokens), "label": cur_label})
            cur_tokens, cur_label = [token], tag[2:]
        elif tag.startswith("I-") and cur_label == tag[2:]:
            cur_tokens.append(token)
        else:
            if cur_tokens:
                entities.append({"text": " ".join(cur_tokens), "label": cur_label})
            cur_tokens, cur_label = [], None
    if cur_tokens:
        entities.append({"text": " ".join(cur_tokens), "label": cur_label})
    return entities
