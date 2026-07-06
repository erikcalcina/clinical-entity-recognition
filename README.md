<p align="center">
  <img src="./docs/assets/logo.svg" alt="Clinical Entity Recognition" width="640">
</p>

<p align="center">
  <b>Clinical entity recognition from medical text under one task formulation and one evaluation.</b><br>
  Reproducibility code for the master thesis <i>Clinical Entity Recognition</i>.
</p>

---

This repository consolidates the code behind the thesis into a single, config-driven
codebase covering three approaches to biomedical named entity recognition, each scored
with the same protocol:

| Approach | What it is | Module |
| --- | --- | --- |
| **LLM** | LoRA fine-tuning with [Unsloth](https://github.com/unslothai/unsloth), plus zero-shot / few-shot prompting | `cer.llm` |
| **GLiNER** | Span extractor, fine-tuned under its native formulation | `cer.gliner` |
| **BERT** | Supervised token classification (BIO) with a domain BERT encoder | `cer.bert` |

Every model emits entities as `(text, label)` pairs and is scored with **Exact F1** and
**Relaxed F1** (thesis Section 3.4). Prompts (Appendix A) and training hyperparameters
(Section 4.4) are reproduced faithfully — they are the defaults in the code, in
`configs/`, and documented per approach below.

## Requirements

- [Python](https://www.python.org/) 3.10+ (`.python-version` pins 3.10).
- A CUDA GPU for training. LLM fine-tuning loads the base model in 4-bit.
- [uv](https://github.com/astral-sh/uv) or `pip` for dependencies.

## Installation

GLiNER and Unsloth are exposed as **separate extras**. Although recent versions overlap
on `transformers`, mixing them in one environment risks subtle conflicts (tokenizers,
accelerate, CUDA wheels). Install one at a time, in its own environment:

```bash
# LLM approach (Unsloth LoRA + prompting)
pip install -e ".[unsloth]"

# GLiNER approach
pip install -e ".[gliner]"

# BERT approach (plain transformers)
pip install -e ".[bert]"
```

## Data

**No datasets are committed.** See [`data/README.md`](./data/README.md) for how to obtain
each corpus (MACCROBAT2020, NCBI Disease Corpus, QUAERO, E3C) and the expected on-disk
layout. The thesis also uses **ICN**, a private Italian clinical corpus that contains
real patient text — it is **excluded** from this repository and must never be committed.

Place each corpus under `data/<dataset>/` (e.g. `data/maccrobat/`) as described there.

## Quickstart

Each approach has a train + evaluate wrapper in `scripts/`. Results are written to
`results/<approach>/…/performance.json`.

### LLM — fine-tuning (Unsloth LoRA)

```bash
# args: <model_id> <dataset:maccrobat|ncbi> [plain|semantic]
bash scripts/train_eval_llm.sh meta-llama/Meta-Llama-3-8B-Instruct maccrobat
```

Defaults (thesis Section 4.4.1): AdamW, lr `2e-4`, `3` epochs, **batch size `4`**,
LoRA rank `16` / alpha `32` / dropout `0.05`, bias none, target modules
`q,k,v,o,gate,down,up`, base model in 4-bit. The exact thesis fine-tuning prompt is
built from `cer.prompts` and passed to the trainer automatically.

### LLM — prompting (zero-shot / few-shot)

```bash
# args: <model_id> <dataset> <zeroshot|fewshot> [plain|semantic]
bash scripts/eval_prompting_llm.sh meta-llama/Meta-Llama-3-8B-Instruct maccrobat fewshot semantic
```

Uses the Appendix A prompts verbatim. MACCROBAT is evaluated per label over the seven
prompt-based labels; `semantic` swaps the short label names for the longer definitions.

### GLiNER

```bash
# args: <gliner_model_id> <dataset>
bash scripts/train_eval_gliner.sh urchade/gliner_large_bio-v0.1 maccrobat
```

Defaults (thesis Section 4.4.2): AdamW, lr `5e-6`, `5` epochs. GLiNER uses the short
entity-type names directly — no semantic definitions.

### BERT

```bash
# args: <bert_model_id> <short_name> <dataset>
bash scripts/train_eval_bert.sh microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext PubMedBERT maccrobat
bash scripts/train_eval_bert.sh emilyalsentzer/Bio_ClinicalBERT BioClinicalBERT maccrobat
```

Defaults (thesis Section 4.4.3): AdamW, lr `5e-5`, `10` epochs, batch `16`. The BIO label
schema is derived from the data and saved with the model; unknown tags raise rather than
being silently dropped. The BERT baseline targets MACCROBAT2020.

## Repository layout

```
clinical-entity-recognition/
├── src/cer/
│   ├── prompts.py            # thesis Appendix A prompts (single source of truth)
│   ├── core/
│   │   ├── data/formatter.py # chat formatting for the LLM approach
│   │   ├── data/convert.py   # GLiNER ⇄ BIO ⇄ span converters
│   │   └── metrics/          # Exact / Relaxed / Overlap F1 (shared scorer)
│   ├── llm/                  # train_unsloth, evaluate_unsloth, evaluate_prompting
│   ├── gliner/               # train, evaluate
│   └── bert/                 # prepare, train, evaluate
├── configs/                  # thesis hyperparameters per approach (YAML)
├── scripts/                  # train + evaluate wrappers per approach
├── data/                     # (empty) — see data/README.md; no data committed
└── docs/assets/logo.svg
```

## Reproducibility notes

- All approaches share one scorer (`cer.core.metrics.NERMetrics`) and report Exact and
  Relaxed F1. LLM decoding is greedy for determinism.
- Prompts (Appendix A) and hyperparameters (Section 4.4) are reproduced faithfully — for
  example the LLM effective batch size is `4`, and the semantic label definitions preserve
  inline case (`mmHg`, `UMLS/MeSH/OMIM`).
- Seeds are fixed (`42`) but results still reflect single runs; the thesis notes that
  repeated runs and per-dataset tuning would tighten the close comparisons (Section 7.2).

## License

BSD 2-Clause — see [`LICENSE`](./LICENSE).
