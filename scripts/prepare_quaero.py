"""Prepare the QUAERO French Medical Corpus into this repo's dataset files.

Source: the ``DrBenchmark/QUAERO`` dataset on the HuggingFace Hub (pre-tokenised with
integer IOB2 tags over the 10 UMLS semantic groups). Downloaded automatically. QUAERO
has two independent sub-corpora — ``emea`` (drug leaflets) and ``medline`` (titles) —
prepared separately; select one with ``--subset``.

    python scripts/prepare_quaero.py --subset emea     --output_dir ./data/quaero_emea
    python scripts/prepare_quaero.py --subset medline  --output_dir ./data/quaero_medline

The official train / validation / test splits are used; validation is merged into the
training file by default (pass ``--drop_validation`` to discard it).
"""

import random
from argparse import ArgumentParser

from cer.core.data.prepare import detokenise, iob_ids_to_spans, write_dataset

# B- tag id -> UMLS semantic group (matching I- id is the B- id + 1).
ID2LABEL = {
    1: "LIVB",
    3: "PROC",
    5: "ANAT",
    7: "DEVI",
    9: "CHEM",
    11: "GEOG",
    13: "PHYS",
    15: "PHEN",
    17: "DISO",
    19: "OBJC",
}


def load_split(subset: str, split: str):
    from datasets import load_dataset

    ds = load_dataset("DrBenchmark/QUAERO", name=subset, split=split, trust_remote_code=True)
    return [
        {"text": detokenise(ex["tokens"]), "entities": iob_ids_to_spans(ex["tokens"], ex["ner_tags"], ID2LABEL)}
        for ex in ds
    ]


def main(args) -> None:
    train = load_split(args.subset, "train")
    if not args.drop_validation:
        train += load_split(args.subset, "validation")
    test = load_split(args.subset, "test")

    random.Random(args.seed).shuffle(train)
    output_dir = args.output_dir or f"./data/quaero_{args.subset}"
    write_dataset(output_dir, train, test)


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare QUAERO (HuggingFace) into repo dataset files.")
    parser.add_argument("--subset", required=True, choices=["emea", "medline"], help="QUAERO sub-corpus")
    parser.add_argument("--output_dir", default=None, help="Output dataset dir (default: ./data/quaero_<subset>)")
    parser.add_argument("--drop_validation", action="store_true", help="Discard the validation split instead of merging into train")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the training set")
    main(parser.parse_args())
