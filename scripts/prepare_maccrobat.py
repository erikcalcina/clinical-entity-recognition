"""Prepare MACCROBAT2020 (raw brat standoff) into this repo's dataset files.

Source: paired ``<id>.txt`` (raw clinical case report) and ``<id>.ann`` (brat text-bound
annotations) that you download from figshare and place in one directory. Documents are
sorted by file name and split at the document level, so the sentence segmentation and the
train/test boundary are identical across the GLiNER, BERT, and LLM approaches.

    python scripts/prepare_maccrobat.py --input_dir /path/to/MACCROBAT2020 --output_dir ./data/maccrobat

Requires nltk for sentence segmentation:  pip install -e ".[data]"
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from cer.core.data.prepare import write_dataset


def format_label(raw_label: str) -> str:
    """``Disease_disorder`` -> ``Disease disorder`` (the thesis 41-label schema)."""
    return raw_label.replace("_", " ").capitalize()


def sentence_splitter():
    """Return an nltk sentence tokenizer, downloading the model if needed."""
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('nltk is required. Install it with:  pip install -e ".[data]"') from exc

    import nltk

    for pkg in ("punkt_tab", "punkt"):  # model name changed across nltk versions
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            break
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # noqa: BLE001
                continue
    return sent_tokenize


def parse_ann(ann_path: Path) -> List[Dict]:
    """Parse text-bound (``T``) annotations. Discontinuous spans collapse to first..last."""
    entities: List[Dict] = []
    with open(ann_path, "r", encoding="utf8") as fa:
        for line in fa:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0].startswith("T"):
                continue
            info = parts[1].replace(";", " ").split()
            label, offsets = info[0], info[1:]
            try:
                start, end = int(offsets[0]), int(offsets[-1])
            except (IndexError, ValueError):
                continue
            entities.append({"label": label, "start": start, "end": end, "text": parts[2]})
    return entities


def load_documents(input_dir: Path) -> List[Dict]:
    """Load ``(text, entities)`` for every doc with both a .txt and .ann, sorted by name."""
    stems = sorted({p.stem for p in input_dir.rglob("*.txt")} & {p.stem for p in input_dir.rglob("*.ann")})
    documents = []
    for stem in stems:
        text = next(input_dir.rglob(f"{stem}.txt")).read_text(encoding="utf8")
        entities = parse_ann(next(input_dir.rglob(f"{stem}.ann")))
        documents.append({"text": text, "entities": entities})
    return documents


def document_to_sentences(doc: Dict, split_sentences) -> List[Dict]:
    """Split a document into span/eval records, one per sentence.

    An entity belongs to the sentence whose character span fully contains it.
    """
    text = doc["text"]
    records, cursor = [], 0
    for sentence in split_sentences(text):
        s_start = text.find(sentence, cursor)
        if s_start == -1:
            continue
        s_end = s_start + len(sentence)
        cursor = s_end
        entities = [
            {"text": e["text"], "label": format_label(e["label"])}
            for e in doc["entities"]
            if s_start <= e["start"] and e["end"] <= s_end
        ]
        records.append({"text": sentence, "entities": entities})
    return records


def main(args) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist")

    split_sentences = sentence_splitter()
    documents = load_documents(input_dir)
    if not documents:
        raise SystemExit(f"No paired .txt/.ann files found under {input_dir}")

    n_train = int(len(documents) * args.train_test_ratio)
    train_records = [r for doc in documents[:n_train] for r in document_to_sentences(doc, split_sentences)]
    test_records = [r for doc in documents[n_train:] for r in document_to_sentences(doc, split_sentences)]

    print(f"MACCROBAT2020: {len(documents)} documents -> {n_train} train / {len(documents) - n_train} test")
    write_dataset(args.output_dir, train_records, test_records)


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare raw MACCROBAT2020 brat standoff into repo dataset files.")
    parser.add_argument("--input_dir", required=True, help="Dir containing MACCROBAT2020 .txt/.ann files")
    parser.add_argument("--output_dir", default="./data/maccrobat", help="Output dataset dir")
    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="Fraction of documents for training")
    main(parser.parse_args())
