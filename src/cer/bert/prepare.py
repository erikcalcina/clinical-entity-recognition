"""Convert GLiNER-format JSON to the BIO format the BERT approach consumes.

- Train file: GLiNER training shape (``tokenized_text`` + word-index ``ner`` spans).
- Test file: span/eval shape (``text`` + ``entities`` strings).

Both are written as ``{"tokens": [...], "ner_tags": ["O", "B-Age", ...]}`` records via
the shared converters in :mod:`cer.core.data.convert`, so the BIO tag strings are
identical to those used everywhere else in the repo.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from cer.core.data.convert import gliner_ner_to_bio, spans_to_bio


def _print_stats(name, records):
    total = sum(len(r["tokens"]) for r in records)
    ent = sum(1 for r in records for tag in r["ner_tags"] if tag != "O")
    print(f"  {name}: {len(records)} sentences, {total} tokens, {ent} entity tokens")


def main(args):
    train_path, test_path = Path(args.data_train_file), Path(args.data_test_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    with open(train_path, encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_path, encoding="utf-8") as f:
        test_raw = json.load(f)

    train_records = [gliner_ner_to_bio(r) for r in train_raw]
    test_records = [spans_to_bio(r) for r in test_raw]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out, test_out = output_dir / "train.ner.json", output_dir / "test.ner.json"

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2, ensure_ascii=False)

    print("Dataset preparation complete:")
    _print_stats("train", train_records)
    _print_stats("test", test_records)
    print(f"  -> {train_out}")
    print(f"  -> {test_out}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert GLiNER JSON to BIO NER format for BERT.")
    parser.add_argument("--data_train_file", type=str, required=True, help="Path to train.gliner.json")
    parser.add_argument("--data_test_file", type=str, required=True, help="Path to test.gliner.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write train.ner.json / test.ner.json")
    main(parser.parse_args())
