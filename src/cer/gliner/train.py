"""Fine-tune a GLiNER span extractor (thesis GLiNER baseline).

Hyperparameter defaults follow the thesis (Section 4.4.2): AdamW, learning rate
5e-6, 5 epochs. The thesis tuned these on ``gliner_large_bio-v0.1`` and applied them
uniformly to every GLiNER variant. GLiNER uses the short entity-type names directly
as label inputs — no semantic definitions (thesis Section 3.3.2).

Input is the GLiNER training shape: ``{"tokenized_text": [...], "ner": [[s, e, label]]}``.
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

import torch

try:
    from gliner import GLiNER
    from gliner.data_processing.collator import SpanDataCollator
    from gliner.training import Trainer, TrainingArguments
except ImportError:
    raise ImportError("GLiNER is not installed. Please install it with: pip install -e .[gliner]")

from cer.core.utils.argument_parsers import str2bool


def get_args():
    parser = ArgumentParser("Fine-tune a GLiNER model")
    parser.add_argument("--train-dataset-file", type=str, required=True, help="Path to the training dataset (JSON)")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Pre-trained GLiNER model name or path")
    parser.add_argument("--model-output-dir", type=str, required=True, help="Output directory for the trained model")
    parser.add_argument("--training-output-dir", type=str, default="models/tmp", help="Trainer working directory")
    parser.add_argument("--train-validation-ratio", type=float, default=0.8, help="Train/validation split ratio")

    # thesis Section 4.4.2
    parser.add_argument("--train-num-epochs", type=int, default=5, help="Number of epochs (thesis: 5)")
    parser.add_argument("--train-learning-rate", type=float, default=5e-6, help="Learning rate (thesis: 5e-6)")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--train-weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train-other-lr", type=float, default=1e-5, help="Learning rate for non-encoder parameters")
    parser.add_argument("--train-other-weight-decay", type=float, default=0.01, help="Weight decay for other params")
    parser.add_argument("--train-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-cpu", type=str2bool, default=False, help="Force CPU")
    return parser.parse_args()


def main(args):
    if not Path(args.train_dataset_file).exists():
        raise FileNotFoundError(f"Train data file {args.train_dataset_file} does not exist")

    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)

    with open(args.train_dataset_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # keep only tokenised records (GLiNER requires the tokenised_text field)
    data = [d for d in data if d.get("tokenized_text")]

    random.shuffle(data)
    if args.train_validation_ratio > 0:
        split = int(len(data) * args.train_validation_ratio)
        train_dataset, valid_dataset = data[:split], data[split:]
    else:
        train_dataset, valid_dataset = data, None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    model = GLiNER.from_pretrained(args.model_name_or_path)
    data_collator = SpanDataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.training_output_dir,
        learning_rate=args.train_learning_rate,
        weight_decay=args.train_weight_decay,
        others_lr=args.train_other_lr,
        others_weight_decay=args.train_other_weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=args.train_num_epochs,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained(args.model_output_dir)


if __name__ == "__main__":
    main(get_args())
