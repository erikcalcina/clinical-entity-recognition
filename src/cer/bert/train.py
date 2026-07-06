"""Fine-tune a BERT encoder for token classification (thesis BERT baseline).

Hyperparameter defaults follow the thesis (Section 4.4.3): AdamW, learning rate
5e-5, 10 epochs, batch size 16. The thesis tuned these on BioClinicalBERT and applied
them uniformly to PubMedBERT; the BERT baseline is evaluated on MACCROBAT2020 only.

The BIO label set is built from the labels present in the training data, and an unknown
tag raises rather than being silently mapped to ``O``, so no entity type can be dropped
unnoticed. The label map is saved into the model config (``id2label``/``label2id``) so
evaluation reconstructs exactly the same schema.
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def build_label_maps(records):
    """Build BIO label lists from the entity labels present in the data.

    Entity labels are sorted for determinism; ``O`` is always id 0.
    """
    entity_labels = sorted({tag[2:] for r in records for tag in r["ner_tags"] if tag != "O"})
    bio_labels = ["O"] + [f"{p}-{lbl}" for lbl in entity_labels for p in ("B", "I")]
    label2id = {label: i for i, label in enumerate(bio_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return bio_labels, label2id, id2label


def tokenize_and_align(examples, tokenizer, label2id, max_length):
    """Sub-word tokenise word-level tokens and align BIO labels.

    Only the first sub-word of each word carries a label; the rest get -100 so the
    loss ignores them. An unknown tag raises ``KeyError`` rather than being silently
    mapped to ``O``.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_label_ids = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev = [], None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev:
                tag = word_labels[word_id]
                if tag not in label2id:
                    raise KeyError(
                        f"BIO tag {tag!r} is not in the label schema {sorted(label2id)}. "
                        "The training data and the label map disagree."
                    )
                label_ids.append(label2id[tag])
            else:
                label_ids.append(-100)
            prev = word_id
        all_label_ids.append(label_ids)
    tokenized["labels"] = all_label_ids
    return tokenized


def build_compute_metrics(id2label):
    """Token-level entity F1 — used only to report progress per epoch. Final reported
    metrics come from :mod:`cer.bert.evaluate` (Exact / Relaxed F1 over spans).
    """

    def compute_metrics(eval_preds):
        logits, label_ids_batch = eval_preds
        predictions = np.argmax(logits, axis=-1)
        tp = fp = fn = 0
        for pred_row, label_row in zip(predictions, label_ids_batch):
            for pred, label in zip(pred_row, label_row):
                if label == -100:
                    continue
                true_tag, pred_tag = id2label[label], id2label[int(pred)]
                true_entity, pred_entity = true_tag != "O", pred_tag != "O"
                match = true_tag == pred_tag
                if true_entity and pred_entity and match:
                    tp += 1
                elif pred_entity and not match:
                    fp += 1
                elif true_entity and not match:
                    fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    return compute_metrics


def main(args):
    data_path = Path(args.data_train_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Train file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bio_labels, label2id, id2label = build_label_maps(raw_data)
    print(f"Label schema: {len(bio_labels)} BIO tags over {(len(bio_labels) - 1) // 2} entity types")

    random.seed(args.seed)
    random.shuffle(raw_data)
    split = int(len(raw_data) * (1.0 - args.val_ratio))
    train_raw, val_raw = raw_data[:split], raw_data[split:]

    train_dataset = Dataset.from_list(train_raw)
    val_dataset = Dataset.from_list(val_raw)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(bio_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    def align(ex):
        return tokenize_and_align(ex, tokenizer, label2id, args.max_length)

    train_tok = train_dataset.map(align, batched=True, remove_columns=["tokens", "ner_tags"])
    val_tok = val_dataset.map(align, batched=True, remove_columns=["tokens", "ner_tags"])

    output_dir = Path(args.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(id2label),
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune a BERT encoder for NER (thesis BERT baseline).")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--data_train_file", type=str, required=True, help="Path to train.ner.json (BIO format)")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of epochs (thesis: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (thesis: 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (thesis: 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of train held out for validation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (also seeds the validation split)")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU")
    main(parser.parse_args())
