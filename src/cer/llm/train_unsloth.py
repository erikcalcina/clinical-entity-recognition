"""LoRA fine-tuning of an LLM with Unsloth (thesis LLM approach).

Hyperparameter defaults follow the thesis (Section 4.4.1): AdamW, learning rate
2e-4, 3 epochs, **batch size 4**, LoRA rank 16 / alpha 32 / dropout 0.05, bias none,
target modules q/k/v/o/gate/down/up, base model loaded in 4-bit.

The thesis was tuned on Meta-Llama-3-8B-Instruct and the configuration applied
uniformly to every LLM variant, so these are the shipped defaults.

The instruction passed via ``--model-system-prompt`` should be the thesis
fine-tuning prompt (see :mod:`cer.prompts`). For MACCROBAT2020, pass::

    python -m cer.prompts ...            # or build with prompts.finetuning_instruction("maccrobat")
"""

import json
import logging
import os
import sys
from argparse import ArgumentParser
from importlib import reload
from pathlib import Path

try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    from trl import SFTConfig, SFTTrainer
except ImportError:
    raise ImportError("Unsloth is not installed. Please install it with: pip install -e .[unsloth]")

from datasets import load_dataset

import cer.core.data.formatter as mfmt
from cer.core.data.convert import detect_entity_key
from cer.core.utils.argument_parsers import str2bool
from cer.prompts import FINETUNING_SYSTEM_PROMPT

# reload custom modules to avoid caching issues
mfmt = reload(mfmt)

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ===============================
# Model helpers
# ===============================


def get_chat_template_name(model_name):
    """Return the unsloth chat-template name for a model identifier."""
    model_name_lower = model_name.lower()

    if "llama-3.2" in model_name_lower:
        return "llama-3.2"
    elif "llama-3.1" in model_name_lower:
        return "llama-3.1"
    elif "llama" in model_name_lower:
        return "llama-3"
    elif "gemma" in model_name_lower and ("3n" in model_name_lower or "3-n" in model_name_lower):
        return "gemma-3n"
    elif "gemma" in model_name_lower:
        return "gemma-3"  # works for gemma-3, medgemma
    elif "qwen" in model_name_lower:
        return "qwen-3"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_train_on_responses_only_params(tokenizer, model_name):
    """Return the instruction/response markers used to mask the loss to responses."""
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        return {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        }
    elif "gemma" in model_name_lower:
        return {"instruction_part": "<start_of_turn>user\n", "response_part": "<start_of_turn>model\n"}
    elif "qwen" in model_name_lower:
        return {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"}
    elif "mistral" in model_name_lower:
        return {"instruction_part": "[INST]", "response_part": "[/INST]"}
    elif hasattr(tokenizer, "chat_template") and "chatml" in (tokenizer.chat_template or "").lower():
        return {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"}
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ===============================
# Main
# ===============================


def get_args():
    parser = ArgumentParser("Fine-tune an LLM with Unsloth (LoRA)")
    # Data
    parser.add_argument("--train-dataset-file", type=str, required=True, help="Path to the training dataset (JSON)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the fine-tuned adapter")

    # Model
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Pre-trained model name or path")
    parser.add_argument("--model-max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--model-load-in-4bit", type=str2bool, default=True, help="Load the base model in 4-bit (thesis)")
    parser.add_argument("--model-load-in-8bit", type=str2bool, default=False, help="Load the base model in 8-bit")
    parser.add_argument("--model-full-finetuning", type=str2bool, default=False, help="Full fine-tuning instead of LoRA")
    parser.add_argument("--model-hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument(
        "--model-system-prompt",
        type=str,
        default=FINETUNING_SYSTEM_PROMPT,
        help="Instruction shown before the medical text. Pass the thesis fine-tuning prompt "
        "(cer.prompts.finetuning_instruction) to reproduce the paper.",
    )
    parser.add_argument("--unique-entities", type=str2bool, default=True, help="Deduplicate entities in the target")

    # LoRA (thesis Section 4.4.1)
    parser.add_argument("--peft-ft-language-layers", type=str2bool, default=True, help="Fine-tune language layers")
    parser.add_argument("--peft-ft-attention-modules", type=str2bool, default=True, help="Fine-tune attention modules")
    parser.add_argument("--peft-ft-mlp-modules", type=str2bool, default=True, help="Fine-tune MLP modules")
    parser.add_argument("--peft-ft-vision-layers", type=str2bool, default=False, help="Fine-tune vision layers")
    parser.add_argument("--peft-rank", type=int, default=16, help="LoRA rank r (thesis: 16)")
    parser.add_argument("--peft-lora-alpha", type=int, default=32, help="LoRA alpha (thesis: 32)")
    parser.add_argument("--peft-lora-dropout", type=float, default=0.05, help="LoRA dropout (thesis: 0.05)")
    parser.add_argument(
        "--peft-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
        help="Comma-separated LoRA target modules (thesis: q/k/v/o/gate/down/up)",
    )

    # Training (thesis Section 4.4.1)
    parser.add_argument("--train-per-device-batch-size", type=int, default=4, help="Per-device batch size (thesis: 4)")
    parser.add_argument(
        "--train-gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Default 1 so the effective batch size equals the thesis value of 4.",
    )
    parser.add_argument("--train-num-epochs", type=int, default=3, help="Number of epochs (thesis: 3)")
    parser.add_argument("--train-learning-rate", type=float, default=2e-4, help="Learning rate (thesis: 2e-4)")
    parser.add_argument("--train-weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train-warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--train-lr-scheduler-type", type=str, default="linear", help="LR scheduler type")
    parser.add_argument("--train-seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main(args):
    if not Path(args.train_dataset_file).exists():
        raise FileNotFoundError(f"Training dataset file not found: {args.train_dataset_file}")

    logger.info(f"Preparing the model and tokenizer for '{args.model_name_or_path}'...")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.model_max_seq_length,
        load_in_4bit=args.model_load_in_4bit,
        load_in_8bit=args.model_load_in_8bit,
        full_finetuning=args.model_full_finetuning,
        device_map="balanced",
        token=args.model_hf_token or os.getenv("HF_TOKEN", None),
    )

    logger.info("Attaching LoRA adapters (bias: none, task type: CAUSAL_LM)...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=args.peft_ft_vision_layers,
        finetune_language_layers=args.peft_ft_language_layers,
        finetune_attention_modules=args.peft_ft_attention_modules,
        finetune_mlp_modules=args.peft_ft_mlp_modules,
        target_modules=args.peft_target_modules.split(","),
        r=args.peft_rank,
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        lora_bias="none",
        random_state=args.train_seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=get_chat_template_name(args.model_name_or_path))

    with open(args.train_dataset_file, "r") as f:
        entity_key = detect_entity_key(json.load(f))

    formatter = mfmt.PromptFormatter(
        input_key="text",
        output_key=entity_key,
        system_prompt=args.model_system_prompt,
        unique_entities=args.unique_entities,
    )

    def formatting_prompts_func(examples):
        return formatter.format_train_example_batch(examples, tokenizer)

    logger.info(f"Loading and formatting '{args.train_dataset_file}'...")
    dataset = load_dataset("json", data_files=args.train_dataset_file, split="train")
    dataset = standardize_data_formats(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.remove_columns([c for c in dataset.features.keys() if c != "text"])
    for idx in range(min(3, len(dataset))):
        logger.info(dataset[idx])

    logger.info("Preparing the trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            dataset_num_proc=1,
            per_device_train_batch_size=args.train_per_device_batch_size,
            gradient_accumulation_steps=args.train_gradient_accumulation_steps,
            warmup_steps=args.train_warmup_steps,
            num_train_epochs=args.train_num_epochs,
            learning_rate=args.train_learning_rate,
            weight_decay=args.train_weight_decay,
            lr_scheduler_type=args.train_lr_scheduler_type,
            optim="adamw_8bit",
            seed=args.train_seed,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
    )

    # mask the loss so only the assistant (JSON) tokens are trained on
    trainer = train_on_responses_only(
        trainer,
        **get_train_on_responses_only_params(tokenizer, args.model_name_or_path),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving the adapter and tokenizer...")
    model_path = Path(args.output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main(get_args())
