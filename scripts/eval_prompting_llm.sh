#!/bin/bash
# Zero-shot / few-shot prompting evaluation of an LLM using the thesis prompts.
# Usage: bash scripts/eval_prompting_llm.sh <model_id> <dataset> <zeroshot|fewshot> [semantic]
set -euo pipefail
trap "exit" INT

MODEL_ID="${1:?model id required}"
DATASET="${2:?dataset required: maccrobat or ncbi}"
REGIME="${3:-zeroshot}"       # zeroshot | fewshot
SEMANTIC="${4:-plain}"        # plain | semantic

TEST_FILE="./data/${DATASET}/test.llm.json"
TAG="$(basename "${MODEL_ID}")_${DATASET}_${REGIME}_${SEMANTIC}"
RESULTS_DIR="./results/prompting/${TAG}"

SEM=false
[ "${SEMANTIC}" = "semantic" ] && SEM=true

python -m cer.llm.evaluate_prompting \
    --model-dir         "${MODEL_ID}" \
    --eval-dataset-file "${TEST_FILE}" \
    --results-dir       "${RESULTS_DIR}" \
    --dataset           "${DATASET}" \
    --regime            "${REGIME}" \
    --semantic          "${SEM}"

echo "Done. Results in ${RESULTS_DIR}/performance.json"
