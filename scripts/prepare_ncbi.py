"""Prepare the NCBI Disease Corpus into this repo's dataset files.

Source: the ``ncbi/ncbi_disease`` dataset on the HuggingFace Hub (pre-tokenised with
integer IOB2 tags; every disease subcategory is collapsed to the single ``DISEASE``
label). Downloaded automatically — no manual files needed.

    python scripts/prepare_ncbi.py --output_dir ./data/ncbi

The official train / validation / test splits are used. Validation is merged into the
training file by default (the pipeline has no separate validation step); pass
``--drop_validation`` to discard it instead.
"""

import random
from argparse import ArgumentParser

from cer.core.data.prepare import detokenise, iob_ids_to_spans, write_dataset

ID2LABEL = {1: "DISEASE"}  # 0=O, 1=B-DISEASE, 2=I-DISEASE


def load_split(split: str):
    from datasets import load_dataset

    ds = load_dataset("ncbi/ncbi_disease", split=split, trust_remote_code=True)
    return [
        {"text": detokenise(ex["tokens"]), "entities": iob_ids_to_spans(ex["tokens"], ex["ner_tags"], ID2LABEL)}
        for ex in ds
    ]


def main(args) -> None:
    train = load_split("train")
    if not args.drop_validation:
        train += load_split("validation")
    test = load_split("test")

    random.Random(args.seed).shuffle(train)
    write_dataset(args.output_dir, train, test)


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare NCBI Disease Corpus (HuggingFace) into repo dataset files.")
    parser.add_argument("--output_dir", default="./data/ncbi", help="Output dataset dir")
    parser.add_argument("--drop_validation", action="store_true", help="Discard the validation split instead of merging into train")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the training set")
    main(parser.parse_args())
