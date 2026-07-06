#!/bin/bash
# Fine-tune and evaluate a GLiNER model with thesis defaults.
# Usage: bash scripts/train_eval_gliner.sh <gliner_model_id> <dataset>
set -euo pipefail
trap "exit" INT

MODEL_ID="${1:?gliner model id required, e.g. urchade/gliner_large_bio-v0.1}"
DATASET="${2:?dataset required, e.g. maccrobat}"

TRAIN_FILE="./data/${DATASET}/train.gliner.json"
TEST_FILE="./data/${DATASET}/test.gliner.json"
TAG="$(basename "${MODEL_ID}")_${DATASET}"
OUT_DIR="./models/gliner/${TAG}"
RESULTS_DIR="./results/gliner/${TAG}"

echo "[train] ${TAG}  (lr 5e-6, epochs 5)"
python -m cer.gliner.train \
    --model-name-or-path "${MODEL_ID}" \
    --train-dataset-file "${TRAIN_FILE}" \
    --model-output-dir   "${OUT_DIR}"

echo "[eval] ${TAG}"
python -m cer.gliner.evaluate \
    --model-dir         "${OUT_DIR}" \
    --eval-dataset-file "${TEST_FILE}" \
    --results-dir       "${RESULTS_DIR}"

echo "Done. Results in ${RESULTS_DIR}/performance.json"
