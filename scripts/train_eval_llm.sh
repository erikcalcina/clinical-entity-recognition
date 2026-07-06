#!/bin/bash
# Fine-tune and evaluate an LLM (Unsloth LoRA) with thesis defaults.
# Usage: bash scripts/train_eval_llm.sh <model_id> <dataset:maccrobat|ncbi> [semantic]
set -euo pipefail
trap "exit" INT

MODEL_ID="${1:?model id required, e.g. meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET="${2:?dataset required: maccrobat or ncbi}"
SEMANTIC="${3:-plain}"   # "semantic" to enable semantic label definitions

TRAIN_FILE="./data/${DATASET}/train.llm.json"
TEST_FILE="./data/${DATASET}/test.llm.json"
TAG="$(basename "${MODEL_ID}")_${DATASET}_${SEMANTIC}"
OUT_DIR="./models/llm/${TAG}"
RESULTS_DIR="./results/llm/${TAG}"

SEM_FLAG=""
[ "${SEMANTIC}" = "semantic" ] && SEM_FLAG="--semantic"

# Build the exact thesis fine-tuning instruction (wrapper + full-schema request).
INSTRUCTION="$(python -m cer.prompts finetune "${DATASET}" ${SEM_FLAG})"

echo "[train] ${TAG}  (lr 2e-4, epochs 3, batch 4, LoRA r16/a32)"
python -m cer.llm.train_unsloth \
    --model-name-or-path "${MODEL_ID}" \
    --train-dataset-file "${TRAIN_FILE}" \
    --output-dir         "${OUT_DIR}" \
    --model-system-prompt "${INSTRUCTION}"

echo "[eval] ${TAG}"
python -m cer.llm.evaluate_unsloth \
    --model-dir          "${OUT_DIR}" \
    --eval-dataset-file  "${TEST_FILE}" \
    --results-dir        "${RESULTS_DIR}" \
    --model-system-prompt "${INSTRUCTION}"

echo "Done. Results in ${RESULTS_DIR}/performance.json"
