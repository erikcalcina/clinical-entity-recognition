#!/bin/bash
# Prepare, fine-tune, and evaluate a BERT encoder with thesis defaults.
# Usage: bash scripts/train_eval_bert.sh <bert_model_id> <short_name> <dataset>
set -euo pipefail
trap "exit" INT

MODEL_ID="${1:?bert model id required}"
NAME="${2:?short name required, e.g. PubMedBERT}"
DATASET="${3:-maccrobat}"

DATA_DIR="./data/${DATASET}"
TAG="${NAME}_${DATASET}_lr5e-5_ep10"
OUT_DIR="./models/bert/${TAG}"
RESULTS_FILE="./results/bert/${TAG}.json"

echo "[prepare] Converting ${DATASET} to BIO format..."
python -m cer.bert.prepare \
    --data_train_file "${DATA_DIR}/train.gliner.json" \
    --data_test_file  "${DATA_DIR}/test.gliner.json" \
    --output_dir      "${DATA_DIR}"

echo "[train] ${TAG}  (lr 5e-5, epochs 10, batch 16)"
python -m cer.bert.train \
    --model_name_or_path "${MODEL_ID}" \
    --data_train_file    "${DATA_DIR}/train.ner.json" \
    --model_output_dir   "${OUT_DIR}"

echo "[eval] ${TAG}"
python -m cer.bert.evaluate \
    --model_name_or_path      "${OUT_DIR}" \
    --data_test_file          "${DATA_DIR}/test.ner.json" \
    --data_original_test_file "${DATA_DIR}/test.gliner.json" \
    --output_file             "${RESULTS_FILE}"

echo "Done. Results in ${RESULTS_FILE}"
