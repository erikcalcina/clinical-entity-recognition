"""Evaluate a fine-tuned BERT encoder on the NER task.

Predictions are decoded from BIO tag sequences back into ``(text, label)`` spans and
scored with the same :class:`~cer.core.metrics.NERMetrics` used for the LLM and GLiNER
approaches, so all three share one scorer (thesis Section 3.4). Gold entities come from
the original span/eval file (original entity strings, no tokenisation artifacts).

This routes scoring through the shared metrics module rather than importing an
evaluator from another project, and the label schema is read from the model config
(``id2label``) saved at training time.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from tqdm import tqdm

import cer.core.metrics as mmts
from cer.core.data.convert import bio_to_entities


def tokenize_for_inference(examples, tokenizer, max_length):
    """Tokenise word-level tokens for inference and record first-sub-word positions.

    ``word_first`` marks, per sub-word, whether it is the first piece of a word, so the
    predicted tag for each word can be read off without needing gold labels.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    first_flags = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        prev, flags = None, []
        for word_id in word_ids:
            flags.append(word_id is not None and word_id != prev)
            prev = word_id
        first_flags.append(flags)
    tokenized["word_first"] = first_flags
    return tokenized


def main(args):
    bio_path = Path(args.data_test_file)
    original_path = Path(args.data_original_test_file)
    if not bio_path.exists():
        raise FileNotFoundError(f"Test file not found: {bio_path}")
    if not original_path.exists():
        raise FileNotFoundError(f"Original test file not found: {original_path}")

    with open(bio_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(original_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    if len(raw_data) != len(original_data):
        raise ValueError(
            f"BIO test file has {len(raw_data)} examples but original has {len(original_data)} — "
            "files must be aligned."
        )

    all_tokens = [ex["tokens"] for ex in raw_data]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    dataset = Dataset.from_list(raw_data)
    tokenized = dataset.map(
        lambda ex: tokenize_for_inference(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=[c for c in dataset.column_names],
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    def collate(batch):
        first_flags = [ex.pop("word_first") for ex in batch]
        encoded = collator(batch)
        return encoded, first_flags

    tokenized.set_format(type="python")
    dataloader = DataLoader(tokenized, batch_size=args.batch_size, collate_fn=collate)

    pred_bio_seqs = []
    for encoded, first_flags in tqdm(dataloader, desc="Evaluating"):
        encoded = {k: torch.as_tensor(v).to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions = logits.argmax(dim=-1).cpu().numpy()
        for pred_row, flags in zip(predictions, first_flags):
            seq = [id2label[int(p)] for p, is_first in zip(pred_row, flags) if is_first]
            pred_bio_seqs.append(seq)

    true_entities = [ex.get("entities", ex.get("labels", [])) for ex in original_data]
    pred_entities = [bio_to_entities(tokens, tags) for tokens, tags in zip(all_tokens, pred_bio_seqs)]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output = [
        {"text": ex["text"], "entities": pred_ents}
        for ex, pred_ents in zip(original_data, pred_entities)
    ]
    with open(output_path.with_suffix(".predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions_output, f, indent=2, ensure_ascii=False)

    metrics = mmts.NERMetrics(metrics=args.eval_metrics.split(","))
    unique_labels = sorted({e["label"] for doc in true_entities for e in doc})
    results = {"total": {}, **{lbl: {} for lbl in unique_labels}}
    for match_type in metrics.metrics:
        p, r, f1 = metrics.evaluate_ner_performance(true_entities, pred_entities, match_type=match_type)
        results["total"][match_type] = {"p": p, "r": r, "f1": f1}
        for label in unique_labels:
            p, r, f1 = metrics.evaluate_ner_performance(true_entities, pred_entities, match_type=match_type, label=label)
            results[label][match_type] = {"p": p, "r": r, "f1": f1}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"num_examples": len(original_data), "metrics": results}, f, indent=2, ensure_ascii=False)

    print(f"\n{'Match type':<10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 42)
    for match_type, m in results["total"].items():
        print(f"{match_type:<10}  {m['p']:>10.4f}  {m['r']:>8.4f}  {m['f1']:>8.4f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a fine-tuned BERT NER model (Exact / Relaxed F1).")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--data_test_file", type=str, required=True, help="Path to test.ner.json (BIO format)")
    parser.add_argument("--data_original_test_file", type=str, required=True, help="Path to test.gliner.json (gold spans)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write evaluation results JSON")
    parser.add_argument("--eval_metrics", type=str, default="exact,relaxed", help="Comma-separated metrics")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU")
    main(parser.parse_args())
