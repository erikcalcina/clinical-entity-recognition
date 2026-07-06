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
    from gliner import GLiNER
except ImportError:
    raise ImportError("GLiNER is not installed. Please install it with: pip install -e .[gliner]")

import cer.core.metrics as mmts
from cer.core.utils.argument_parsers import str2bool

mmts = reload(mmts)

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # override any existing handlers
)
logger = logging.getLogger(__name__)

# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Evaluate a GLiNER model")
    # Data
    parser.add_argument("--eval-dataset-file", type=str, help="The path to the evaluation data file")
    parser.add_argument("--results-dir", type=str, help="The path to the output file")

    # Model
    parser.add_argument("--model-dir", type=str, help="The path to the pre-trained model or the model name")

    # Validation parameters
    parser.add_argument(
        "--eval-threshold", type=float, default=0.5, help="The threshold for the prediction (default is 0.5)"
    )
    parser.add_argument("--eval-metrics", type=str, default="exact,relaxed", help="Comma-separated metrics")
    parser.add_argument("--use-cpu", type=str2bool, default=False, help="Whether to use the CPU (default is GPU)")
    return parser.parse_args()


def main(args):
    if not Path(args.eval_dataset_file).exists():
        raise FileNotFoundError(f"Test data file {args.eval_dataset_file} does not exist")

    with open(args.eval_dataset_file, "r", encoding="utf8") as f:
        test_dataset = json.load(f)

    # load the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    model = GLiNER.from_pretrained(args.model_dir, load_tokenizer=True, local_files_only=True)
    model.to(device)

    # change the labels only to include those that you want to evaluate
    unique_labels = list(set([ent["label"] for example in test_dataset for ent in example["entities"]]))

    # get the true and predicted entities
    true_pred_ents = {
        "true_ents": [],
        "pred_ents": [],
    }

    start_time = time.time()
    for example in tqdm(test_dataset, desc="Processing examples"):
        true_pred_ents["true_ents"].append([ent for ent in example["entities"] if ent["label"] in unique_labels])
        with torch.no_grad():
            _pred_ents = model.predict_entities(example["text"], unique_labels, threshold=args.eval_threshold)
        true_pred_ents["pred_ents"].append(_pred_ents)

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(test_dataset)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the true and predicted entities
    with open(results_dir / "true_pred_entities.json", "w") as f:
        json.dump(true_pred_ents, f, ensure_ascii=False)

    # =================================
    # Evaluate the model
    # =================================

    logger.info("Computing the performance...")

    performance = {
        "num_examples": len(test_dataset),
        "avg_inference_time": avg_inference_time,
        "metrics": {
            "total": {},
            **{lbl: {} for lbl in unique_labels},
        },
    }

    eval_metrics = args.eval_metrics.split(",")
    metrics = mmts.NERMetrics(metrics=eval_metrics)
    for match_type in metrics.metrics:
        precision, recall, f1 = metrics.evaluate_ner_performance(
            true_pred_ents["true_ents"],
            true_pred_ents["pred_ents"],
            match_type=match_type,
        )
        performance["metrics"]["total"][match_type] = {"p": precision, "r": recall, "f1": f1}

        for label in unique_labels:
            precision, recall, f1 = metrics.evaluate_ner_performance(
                true_pred_ents["true_ents"],
                true_pred_ents["pred_ents"],
                match_type=match_type,
                label=label,
            )
            performance["metrics"][label][match_type] = {"p": precision, "r": recall, "f1": f1}

    # save the true and predicted entities
    with open(results_dir / "performance.json", "w") as f:
        json.dump(performance, f, ensure_ascii=False)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    args = get_args()
    main(args)
