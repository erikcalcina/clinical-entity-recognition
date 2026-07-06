"""Turn per-sentence span records into this repo's on-disk dataset files.

Every ``scripts/prepare_*.py`` converter has the same job: read its source corpus,
emit two lists of **span/eval records** (``{"text": ..., "entities": [{"text", "label"}]}``
— the ``labels`` key is also accepted), one for train and one for test, then hand them
to :func:`write_dataset`. This module owns everything downstream of that so the four
output files are produced identically for every dataset::

    data/<dataset>/
    ├── train.gliner.json   # GLiNER training shape (tokenized_text + word-index ner spans)
    ├── test.gliner.json    # span/eval shape
    ├── train.llm.json      # span/eval shape
    └── test.llm.json       # span/eval shape

The GLiNER training shape is derived from the span records with :func:`spans_to_gliner`,
which reuses :func:`cer.core.data.convert.spans_to_bio` so the word tokenisation and
entity-to-token alignment match the BERT pipeline exactly.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from cer.core.data.convert import tokenize_with_offsets


def detokenise(tokens: List[str]) -> str:
    """Join word tokens back into readable text (PubMed / clinical spacing rules).

    Used by the token-and-BIO HuggingFace sources (NCBI, QUAERO) whose examples are
    pre-tokenised. Mirrors the spacing normalisation used to build the thesis corpora.
    """
    sent = " ".join(tokens)
    sent = re.sub(r"\s*\+\s*", "+", sent)
    sent = re.sub(r"\s*/\s*", "/", sent)
    sent = re.sub(r"\s*([-‑–])\s*", "‑", sent)
    sent = re.sub(r"\s+([,.;:%])", r"\1", sent)
    sent = re.sub(r"\(\s+", "(", sent)
    sent = re.sub(r"\s+\)", ")", sent)
    sent = re.sub(r"\s+'\s+", "'", sent)
    return sent.strip()


def iob_ids_to_spans(tokens: List[str], tag_ids: List[int], id2label: Dict[int, str]) -> List[Dict]:
    """Convert integer IOB2 tag ids to ``[{"text", "label"}, ...]`` entity spans.

    Assumes the HuggingFace convention where odd ids are ``B-`` and the following even
    id is the matching ``I-`` (``B = k`` continues while the tag equals ``k + 1``).
    ``id2label`` maps each ``B-`` id to its output label string.
    """
    spans: List[Dict] = []
    i = 0
    while i < len(tag_ids):
        tag = tag_ids[i]
        if tag % 2 != 0:  # B- tag
            start = i
            i += 1
            while i < len(tag_ids) and tag_ids[i] == tag + 1:  # matching I- tag
                i += 1
            spans.append({"text": detokenise(tokens[start:i]), "label": id2label[tag]})
        else:
            i += 1
    return spans


def normalise_entities(record: Dict) -> Dict:
    """Return a span/eval record with entities under the ``entities`` key.

    Accepts source records that store their entity list under either ``entities`` or
    ``labels`` and keeps only ``text`` / ``label`` per entity.
    """
    entities = record.get("entities", record.get("labels", []))
    return {
        "text": record["text"],
        "entities": [{"text": e["text"], "label": e["label"]} for e in entities],
    }


def spans_to_gliner(record: Dict) -> Dict:
    """Convert a span/eval record to the GLiNER training shape.

    Whitespace-tokenises the text and maps each entity's character span to an inclusive
    word-index span ``[start, end, label]``. Repeated entity surfaces are matched left to
    right. Overlapping and nested entities are **kept** (GLiNER's span format allows them);
    the BERT pipeline collapses overlaps later when it derives BIO from these spans.
    """
    text = record["text"]
    tokens, offsets = tokenize_with_offsets(text)
    entities = record.get("entities", record.get("labels", []))

    def first_occurrence(entry: Dict) -> int:
        idx = text.find(entry["text"])
        return idx if idx != -1 else len(text)

    ner: List = []
    search_pos: Dict[str, int] = {}
    for entry in sorted(entities, key=first_occurrence):
        surface, label = entry["text"], entry["label"]
        char_start = text.find(surface, search_pos.get(surface, 0))
        if char_start == -1:
            char_start = text.find(surface)  # surface repeats fewer times than annotated
            if char_start == -1:
                continue
        char_end = char_start + len(surface)
        search_pos[surface] = char_end

        # A word belongs to the entity if its character span overlaps the entity's.
        covered = [i for i, (ts, te) in enumerate(offsets) if ts < char_end and te > char_start]
        if covered:
            ner.append([covered[0], covered[-1], label])
    return {"tokenized_text": tokens, "ner": ner}


def write_dataset(output_dir, train_records: List[Dict], test_records: List[Dict]) -> None:
    """Write the four dataset files this repo expects into ``output_dir``.

    ``train_records`` / ``test_records`` are span/eval records (``entities`` or
    ``labels`` key). The train GLiNER file uses the training shape; the other three
    use the span/eval shape.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_span = [normalise_entities(r) for r in train_records]
    test_span = [normalise_entities(r) for r in test_records]

    _dump(output_dir / "train.gliner.json", [spans_to_gliner(r) for r in train_span])
    _dump(output_dir / "test.gliner.json", test_span)
    _dump(output_dir / "train.llm.json", train_span)
    _dump(output_dir / "test.llm.json", test_span)

    def _n_ent(records):
        return sum(len(r["entities"]) for r in records)

    print(
        f"Wrote {output_dir}/ : "
        f"train {len(train_span)} sentences / {_n_ent(train_span)} entities, "
        f"test {len(test_span)} sentences / {_n_ent(test_span)} entities"
    )


def _dump(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
