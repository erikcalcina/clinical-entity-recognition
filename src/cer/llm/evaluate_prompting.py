"""Zero-shot / few-shot prompting evaluation for the LLM approach.

Reproduces the thesis prompt-based experiments (Section 5.1). Prompts come from
:mod:`cer.prompts` (Appendix A). MACCROBAT2020 is evaluated per label over the seven
prompt-based labels and the predictions are merged per sentence; NCBI uses its single
``DISEASE`` label. The ``--semantic`` flag switches plain descriptions for the longer
semantic definitions.

Generation runs through Unsloth so the same environment serves prompting and
fine-tuning. Reported metrics are Exact and Relaxed F1 (thesis Section 3.4).
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
from cer import prompts

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
    parser = ArgumentParser("Evaluate an LLM under zero-shot / few-shot prompting")
    parser.add_argument("--eval-dataset-file", type=str, required=True, help="Path to the evaluation dataset (JSON)")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to write results")
    parser.add_argument("--model-dir", type=str, required=True, help="Model ID or local path")

    parser.add_argument("--dataset", type=str, required=True, choices=["maccrobat", "ncbi"], help="Which prompt set")
    parser.add_argument("--regime", type=str, default="zeroshot", choices=["zeroshot", "fewshot"], help="Prompt regime")
    parser.add_argument("--semantic", type=str2bool, default=False, help="Use semantic label definitions")

    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--model-load-in-4bit", type=str2bool, default=True, help="Load the model in 4-bit")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--eval-metrics", type=str, default="exact,relaxed", help="Comma-separated metrics")
    return parser.parse_args()


def _label_keys(dataset):
    """Prompt-label keys to iterate over for a dataset."""
    return list(prompts.MACCROBAT_PROMPT_LABELS.keys()) if dataset == "maccrobat" else [None]


def _build_messages(dataset, regime, label_key, semantic, text):
    if regime == "fewshot":
        return prompts.fewshot_messages(dataset, text, label_key=label_key, semantic=semantic)
    return prompts.zeroshot_messages(dataset, text, label_key=label_key, semantic=semantic)


def main(args):
    if not Path(args.eval_dataset_file).exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {args.eval_dataset_file}")

    logger.info(f"Loading the model '{args.model_dir}'...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.model_max_seq_length,
        load_in_4bit=args.model_load_in_4bit,
        device_map="balanced",
    )
    model = FastModel.for_inference(model)
    text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    with open(args.eval_dataset_file, "r") as f:
        test_dataset = json.load(f)

    formatter = mfmt.PromptFormatter(input_key="text", output_key="entities", unique_entities=True)

    # gold entities per sentence (restricted to the prompt-based label set for MACCROBAT)
    prompt_labels = set(prompts.MACCROBAT_PROMPT_LABELS.values()) if args.dataset == "maccrobat" else {prompts.NCBI_LABEL}
    entity_key = detect_entity_key(test_dataset)
    true_ents = []
    for example in test_dataset:
        gold = [formatter.format_entities(e) for e in example[entity_key] if e["label"] in prompt_labels]
        true_ents.append(formatter.get_unique_entities(gold))

    # build the full set of (sentence index, label key, rendered prompt) requests
    requests = []
    for si, example in enumerate(test_dataset):
        for label_key in _label_keys(args.dataset):
            messages = _build_messages(args.dataset, args.regime, label_key, args.semantic, example["text"])
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            requests.append((si, rendered))

    pred_ents = [[] for _ in test_dataset]
    model_settings = {"do_sample": False}
    max_new_tokens = min(args.model_max_seq_length // 4, 256)
    batches = [requests[i : i + args.eval_batch_size] for i in range(0, len(requests), args.eval_batch_size)]

    start_time = time.time()
    for batch in tqdm(batches, desc="Generating the model outputs"):
        inputs = text_tokenizer(
            [rendered for _, rendered in batch],
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=args.model_max_seq_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **model_settings)
        for idx, ((si, _rendered), output) in enumerate(zip(batch, outputs)):
            output = output[inputs.input_ids[idx].shape[0] :]
            output = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            entities = formatter.extract_entities_from_text(output)
            # keep only non-empty entity texts (empty string signals "not present")
            pred_ents[si].extend(formatter.format_entities(e) for e in entities if e.get("text"))
    avg_inference_time = (time.time() - start_time) / max(len(requests), 1)

    pred_ents = [formatter.get_unique_entities(p) for p in pred_ents]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "true_pred_entities.json", "w") as f:
        json.dump({"true_ents": true_ents, "pred_ents": pred_ents}, f, ensure_ascii=False)

    logger.info("Computing the performance...")
    unique_labels = sorted(prompt_labels)
    performance = {
        "num_examples": len(test_dataset),
        "regime": args.regime,
        "semantic": bool(args.semantic),
        "avg_inference_time": avg_inference_time,
        "metrics": {"total": {}, **{lbl: {} for lbl in unique_labels}},
    }
    metrics = mmts.NERMetrics(metrics=args.eval_metrics.split(","))
    for match_type in metrics.metrics:
        p, r, f1 = metrics.evaluate_ner_performance(true_ents, pred_ents, match_type=match_type)
        performance["metrics"]["total"][match_type] = {"p": p, "r": r, "f1": f1}
        for label in unique_labels:
            p, r, f1 = metrics.evaluate_ner_performance(true_ents, pred_ents, match_type=match_type, label=label)
            performance["metrics"][label][match_type] = {"p": p, "r": r, "f1": f1}

    with open(results_dir / "performance.json", "w") as f:
        json.dump(performance, f, ensure_ascii=False)
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main(get_args())
