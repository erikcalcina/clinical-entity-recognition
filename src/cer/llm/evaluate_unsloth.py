"""Evaluate a fine-tuned (or base) LLM with Unsloth on the NER task.

Greedy decoding is used for reproducibility. The instruction passed via
``--model-system-prompt`` must match the one used for training (the thesis
fine-tuning prompt from :mod:`cer.prompts`). Reported metrics are Exact and
Relaxed F1 (thesis Section 3.4); ``overlap`` is also available.
"""

import json
import logging
import sys
import time
from argparse import ArgumentParser
from importlib import reload
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from unsloth import FastModel
except ImportError:
    raise ImportError("Unsloth is not installed. Please install it with: pip install -e .[unsloth]")

import cer.core.data.formatter as mfmt
import cer.core.metrics as mmts
from cer.core.data.convert import detect_entity_key
from cer.core.utils.argument_parsers import str2bool
from cer.prompts import FINETUNING_SYSTEM_PROMPT

mfmt = reload(mfmt)
mmts = reload(mmts)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser("Evaluate an LLM with Unsloth")
    parser.add_argument("--eval-dataset-file", type=str, required=True, help="Path to the evaluation dataset (JSON)")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to write results")

    parser.add_argument("--model-dir", type=str, required=True, help="Fine-tuned adapter path or base model ID")
    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--model-load-in-4bit", type=str2bool, default=True, help="Load the model in 4-bit")
    parser.add_argument("--model-load-in-8bit", type=str2bool, default=False, help="Load the model in 8-bit")
    parser.add_argument(
        "--model-system-prompt",
        type=str,
        default=FINETUNING_SYSTEM_PROMPT,
        help="Instruction shown before the medical text. Must match the training prompt.",
    )

    parser.add_argument("--eval-batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--eval-unique-entities", type=str2bool, default=True, help="Deduplicate gold entities")
    parser.add_argument("--eval-metrics", type=str, default="exact,relaxed", help="Comma-separated metrics")
    return parser.parse_args()


def main(args):
    model_is_local = Path(args.model_dir).exists()
    if not model_is_local and "/" not in args.model_dir:
        raise FileNotFoundError(f"Model directory not found and does not look like a HF model ID: {args.model_dir}")
    if not Path(args.eval_dataset_file).exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {args.eval_dataset_file}")

    logger.info(f"Loading the model '{args.model_dir}'...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.model_max_seq_length,
        load_in_4bit=args.model_load_in_4bit,
        load_in_8bit=args.model_load_in_8bit,
        device_map="balanced",
    )
    model = FastModel.for_inference(model)

    with open(args.eval_dataset_file, "r") as f:
        test_dataset = json.load(f)

    formatter = mfmt.PromptFormatter(
        input_key="text",
        output_key="entities",
        system_prompt=args.model_system_prompt,
        unique_entities=args.eval_unique_entities,
    )

    entity_key = detect_entity_key(test_dataset)
    examples = []
    for example in test_dataset:
        txt = formatter.format_test_example(example, tokenizer)
        entities = [formatter.format_entities(e) for e in example[entity_key]]
        if args.eval_unique_entities:
            entities = formatter.get_unique_entities(entities)
        examples.append({"text": txt, "entities": entities})

    # greedy decoding: deterministic and reproducible for structured-output eval
    model_settings = {"do_sample": False}
    max_new_tokens = min(args.model_max_seq_length // 4, 256)
    example_batches = [examples[i : i + args.eval_batch_size] for i in range(0, len(examples), args.eval_batch_size)]

    true_pred_ents = {"true_ents": [], "pred_ents": []}

    start_time = time.time()
    for batch in tqdm(example_batches, desc="Generating the model outputs"):
        # use the underlying text tokenizer directly to avoid an unsloth Gemma3
        # processor patch that drops padding=True when batching text-only inputs
        text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        inputs = text_tokenizer(
            [ex["text"] for ex in batch],
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=args.model_max_seq_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **model_settings)
        for idx, (example, output) in enumerate(zip(batch, outputs)):
            output = output[inputs.input_ids[idx].shape[0] :]
            output = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output = formatter.extract_entities_from_text(output)
            true_pred_ents["true_ents"].append(example["entities"])
            true_pred_ents["pred_ents"].append(output)
    avg_inference_time = (time.time() - start_time) / len(examples)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "true_pred_entities.json", "w") as f:
        json.dump(true_pred_ents, f, ensure_ascii=False)

    logger.info("Computing the performance...")
    unique_labels = sorted(set(lbl["label"] for e in examples for lbl in e["entities"]))
    performance = {
        "num_examples": len(examples),
        "avg_inference_time": avg_inference_time,
        "metrics": {"total": {}, **{lbl: {} for lbl in unique_labels}},
    }

    metrics = mmts.NERMetrics(metrics=args.eval_metrics.split(","))
    for match_type in metrics.metrics:
        p, r, f1 = metrics.evaluate_ner_performance(
            true_pred_ents["true_ents"], true_pred_ents["pred_ents"], match_type=match_type
        )
        performance["metrics"]["total"][match_type] = {"p": p, "r": r, "f1": f1}
        for label in unique_labels:
            p, r, f1 = metrics.evaluate_ner_performance(
                true_pred_ents["true_ents"], true_pred_ents["pred_ents"], match_type=match_type, label=label
            )
            performance["metrics"][label][match_type] = {"p": p, "r": r, "f1": f1}

    with open(results_dir / "performance.json", "w") as f:
        json.dump(performance, f, ensure_ascii=False)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main(get_args())
