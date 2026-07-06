"""Prepare the E3C corpus (Layer 1, fully manual) into this repo's dataset files.

Source: the ``bio-datasets/e3c`` dataset on the HuggingFace Hub, which stores each
clinical case as ``passages`` (sentences with character offsets) plus document-level
``entities``. Downloaded automatically. Each language is a separate corpus, fine-tuned
and evaluated on its own; select one or more with ``--langs`` (default: all five).

    python scripts/prepare_e3c.py --langs en it              # writes ./data/e3c_en, ./data/e3c_it
    python scripts/prepare_e3c.py --langs all --output_root ./data

E3C ships without an official train/test split, so documents are shuffled with a fixed
seed and split by ``--train_test_ratio``. The seed makes the split reproducible (the
original thesis notebook shuffled unseeded).
"""

import random
from argparse import ArgumentParser
from typing import Dict, List

from cer.core.data.prepare import write_dataset

ALL_LANGS = ["en", "it", "fr", "es", "eu"]


def _span_boundaries(offsets):
    """Collapse one- or multi-span offset lists to a single ``(start, end)``."""
    if isinstance(offsets[0], list):  # discontinuous -> list of [start, end]
        starts, ends = zip(*offsets)
        return min(starts), max(ends)
    return tuple(offsets)


def record_to_sentences(record: Dict) -> List[Dict]:
    """Split one E3C document into span/eval records, one per passage (sentence)."""
    sentences = []
    for passage in record["passages"]:
        p_start, p_end = passage["offsets"]
        entities = []
        for ent in record["entities"]:
            e_start, e_end = _span_boundaries(ent["offsets"])
            if p_start <= e_start and e_end <= p_end:
                entities.append({"text": ent["text"], "label": ent["type"]})
        sentences.append({"text": passage["text"], "entities": entities})
    return sentences


def prepare_language(ds, lang: str, output_root: str, ratio: float, seed: int) -> None:
    records = [rec for doc in ds[f"{lang}.layer1"] for rec in record_to_sentences(doc)]
    random.Random(seed).shuffle(records)
    split_idx = int(ratio * len(records))
    print(f"E3C {lang}: {len(records)} sentences -> {split_idx} train / {len(records) - split_idx} test")
    write_dataset(f"{output_root.rstrip('/')}/e3c_{lang}", records[:split_idx], records[split_idx:])


def main(args) -> None:
    from datasets import load_dataset

    langs = ALL_LANGS if args.langs == ["all"] else args.langs
    ds = load_dataset("bio-datasets/e3c", trust_remote_code=True)
    for lang in langs:
        prepare_language(ds, lang, args.output_root, args.train_test_ratio, args.seed)


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare E3C Layer 1 (HuggingFace) into repo dataset files.")
    parser.add_argument("--langs", nargs="+", default=["all"], choices=ALL_LANGS + ["all"], help="Languages to prepare")
    parser.add_argument("--output_root", default="./data", help="Root under which ./e3c_<lang> dirs are written")
    parser.add_argument("--train_test_ratio", type=float, default=0.8, help="Fraction of sentences for training")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split")
    main(parser.parse_args())
