# Data

**No datasets are committed to this repository.** This directory is a placeholder;
`.gitignore` excludes everything here except this README. You must obtain each corpus
from its original source and place it here in the expected layout.

> ⚠️ **Private data.** The thesis also reports results on **ICN**, a private corpus of
> Italian clinical notes from an institutional partner. That corpus contains real
> patient text and is **not distributable** — it is deliberately excluded from this
> repository and must never be committed to a public remote.

## Expected layout

Each dataset lives in its own subdirectory. The training/evaluation scripts expect:

```
data/
└── <dataset>/                     # e.g. maccrobat, ncbi
    ├── train.gliner.json          # GLiNER training shape (used by GLiNER + BERT prepare)
    ├── test.gliner.json           # span/eval shape (gold spans for GLiNER + BERT eval)
    ├── train.llm.json             # span/eval shape (LLM fine-tuning)
    └── test.llm.json              # span/eval shape (LLM evaluation)
```

`cer.bert.prepare` derives `train.ner.json` / `test.ner.json` (BIO format) from the
`*.gliner.json` files, so you do not create those by hand.

## File formats

**GLiNER training shape** (`train.gliner.json`) — word-tokenised text with word-index
spans:

```json
{"tokenized_text": ["A", "31-year-old", "man", "..."],
 "ner": [[1, 1, "Age"], [2, 2, "Sex"]]}
```

**Span / eval shape** (`test.gliner.json`, `*.llm.json`) — raw text with entity strings.
The entity list may be under either the `entities` or the `labels` key; both are
accepted (the code auto-detects):

```json
{"text": "A 31-year-old man developed diabetes insipidus ...",
 "entities": [{"text": "31-year-old", "label": "Age"},
              {"text": "man", "label": "Sex"}]}
```

## Corpora and how to obtain them

The thesis uses five biomedical corpora. Prompt-based experiments use MACCROBAT2020 and
NCBI; the others are fine-tuning only.

| Dataset | Language(s) | Source |
| --- | --- | --- |
| **MACCROBAT2020** | English | figshare — "MACCROBAT" clinical case reports (41 entity types, brat standoff). |
| **NCBI Disease Corpus** | English | NCBI — PubMed abstracts annotated for disease mentions (collapse all subcategories to `DISEASE`). |
| **QUAERO French Medical Corpus** | French | QUAERO — MEDLINE titles + EMEA documents (fine-tuning only). |
| **E3C** | en / fr / it / es / eu | European Clinical Case Corpus — use Layer 1 (fully manual). Each language is fine-tuned and evaluated separately. |
| **ICN** | Italian | Private institutional corpus. **Not distributable; excluded from this repo.** |

MACCROBAT uses the 41-label schema (see `cer.prompts.MACCROBAT_LABELS`); NCBI collapses
every disease subcategory into the single `DISEASE` label.

## Preparing the corpora

`scripts/prepare_*.py` convert each corpus into the four files above. Every script writes
one deterministic split, so the sentence segmentation and the train/test boundary are
identical across the GLiNER, BERT, and LLM approaches by construction — the comparability
requirement from thesis Section 3.4. Install the extra dependencies first:

```bash
uv pip install -e ".[data]"  # nltk (MACCROBAT); `datasets` is already a core dependency
```

| Corpus | Command | Source |
| --- | --- | --- |
| MACCROBAT2020 | `python scripts/prepare_maccrobat.py --input_dir <raw> --output_dir data/maccrobat` | Local brat `.txt`/`.ann` you download from figshare |
| NCBI | `python scripts/prepare_ncbi.py --output_dir data/ncbi` | HuggingFace `ncbi/ncbi_disease` (auto-download) |
| QUAERO | `python scripts/prepare_quaero.py --subset emea` (and `--subset medline`) | HuggingFace `DrBenchmark/QUAERO` (auto-download) |
| E3C | `python scripts/prepare_e3c.py --langs all` | HuggingFace `bio-datasets/e3c`, Layer 1 (auto-download) |

The HuggingFace-based scripts download the corpus for you; only MACCROBAT needs files on
disk. QUAERO's two sub-corpora and E3C's five languages are written to separate dataset
directories (`data/quaero_emea`, `data/e3c_it`, …), each fine-tuned and evaluated on its own.

Shared conversion logic lives in `cer.core.data.prepare`: each script produces per-sentence
span records and hands them to `write_dataset`, which derives the GLiNER training shape and
writes all four files. Entities are matched to word tokens by string search (the same method
`cer.core.data.convert` uses for BERT/eval), so a small fraction of short, ambiguous surfaces
may align imperfectly — training and evaluation share this behaviour, keeping scores comparable.

**ICN** is private and has no converter here; prepare it into the same four-file layout
separately and never commit it.
