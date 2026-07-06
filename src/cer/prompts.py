"""Prompts used in the thesis experiments — reproduced from Appendix A.

This module is the single source of truth for every prompt string. It covers the
three regimes of the LLM approach:

- **Zero-shot** — per-label extraction request, no examples.
- **Few-shot** — the same request with three worked examples (two positive, one
  negative) prepended before the target sentence.
- **Fine-tuning** — a single-pass request listing the full label schema.

Each regime has a **plain** and a **semantic** variant. In the semantic variant the
short label name is replaced by a longer definition (thesis Section 3.2.3).

Conventions:

- The shared instruction wrapper is reproduced from Appendix A verbatim.
- Semantic definitions preserve inline case exactly as written (e.g. ``mmHg``,
  ``UMLS/MeSH/OMIM``); only the leading character is lower-cased so the definition
  reads naturally after "Please extract ...".
- The fine-tuning label schema uses the dataset's own label strings (e.g.
  ``"Sign symptom"``), so the JSON labels the model produces match the gold data.
"""

import json
from typing import Dict, List

# ===============================================================================
# Shared instruction wrappers (thesis Appendix A)
# ===============================================================================

#: Wrapper for the zero-shot and few-shot regimes.
ZEROSHOT_FEWSHOT_SYSTEM_PROMPT = (
    "You are tasked with performing Named Entity Recognition on medical text. Your goal is to "
    "extract specific entities and return them in a structured JSON format. If the entity is not "
    'present in the text, include an empty string ("") for the text field. Ensure the JSON output '
    "is accurate and adheres to the described format."
)

#: Wrapper for the fine-tuning regime. Drops the empty-output sentence, because
#: absence is learned from the training data rather than instructed.
FINETUNING_SYSTEM_PROMPT = (
    "You are tasked with performing Named Entity Recognition on medical text. Your goal is to "
    "extract specific entities and return them in a structured JSON format. Ensure the JSON output "
    "is accurate and adheres to the described format."
)

# ===============================================================================
# Label schemas
# ===============================================================================

#: The 41-label MACCROBAT2020 fine-tuning schema, in the thesis' listing order
#: (alphabetical), using the exact label strings found in the corpus.
MACCROBAT_LABELS: List[str] = [
    "Activity",
    "Administration",
    "Age",
    "Area",
    "Biological attribute",
    "Biological structure",
    "Clinical event",
    "Color",
    "Coreference",
    "Date",
    "Detailed description",
    "Diagnostic procedure",
    "Disease disorder",
    "Distance",
    "Dosage",
    "Duration",
    "Family history",
    "Frequency",
    "Height",
    "History",
    "Lab value",
    "Mass",
    "Medication",
    "Nonbiological location",
    "Occupation",
    "Other entity",
    "Other event",
    "Outcome",
    "Personal background",
    "Qualitative concept",
    "Quantitative concept",
    "Severity",
    "Sex",
    "Shape",
    "Sign symptom",
    "Subject",
    "Texture",
    "Therapeutic procedure",
    "Time",
    "Volume",
    "Weight",
]

#: The seven labels used in the prompt-based (zero-shot / few-shot) MACCROBAT2020
#: experiments, mapped to the exact label string used in the JSON output.
MACCROBAT_PROMPT_LABELS: Dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "biological_structure": "Biological structure",
    "sign_symptom": "Sign symptom",
    "diagnostic_procedure": "Diagnostic procedure",
    "lab_value": "Lab value",
    "detailed_description": "Detailed description",
}

NCBI_LABEL = "DISEASE"

# ===============================================================================
# Plain label descriptions (thesis Appendix A.1.1 / A.3.1)
# ===============================================================================

MACCROBAT_PLAIN_DESCRIPTIONS: Dict[str, str] = {
    "age": "The age of the patient.",
    "sex": "The sex or gender of the patient.",
    "biological_structure": "Any part of the body, from the cellular level to general areas.",
    "sign_symptom": "Any symptom or clinical finding.",
    "diagnostic_procedure": "Any procedure done primarily in order to obtain more information.",
    "lab_value": "Any result of a laboratory test or a diagnostic result, including any units or values present",
    "detailed_description": "Any detail of an event or other entity.",
}

# ===============================================================================
# Semantic label definitions (thesis Appendix A.1.2 / A.3.2)
# ===============================================================================

MACCROBAT_SEMANTIC_DESCRIPTIONS: Dict[str, str] = {
    "age": (
        "The duration of time a patient has lived, expressed numerically (e.g., '65-year-old', "
        "'20 years old') or categorically (e.g., 'newborn', 'teenage'), representing their age at "
        "the time of presentation."
    ),
    "sex": (
        "Explicit mention of a patient's biological sex, indicated by terms such as 'male', 'female', "
        "'man', or 'woman', reflecting the classification based on reproductive anatomy and genetic "
        "attributes."
    ),
    "biological_structure": (
        "Any explicitly stated anatomical term referring to a specific part of the human body, ranging "
        "from macroscopic structures like organs and limbs to microscopic entities such as cells and "
        "tissues, including precise locations and directional descriptors."
    ),
    "sign_symptom": (
        "Any explicitly stated indication of a patient's abnormal condition, encompassing subjective "
        "experiences such as pain or dizziness, objective findings like fever or rash, and descriptive "
        "phrases detailing clinical abnormalities, including those identified through diagnostic "
        "procedures or imaging."
    ),
    "diagnostic_procedure": (
        "Any explicitly stated or contextually implied name of a medical procedure performed to obtain "
        "diagnostic information about a patient or symptom, including formal procedure names, physical "
        "examinations, imaging studies, laboratory tests, and diagnostic screens, regardless of whether "
        "the text states the procedure result; includes both standalone and component procedures."
    ),
    "lab_value": (
        "Any explicitly stated result of a diagnostic procedure, encompassing numerical measurements "
        "with or without units (e.g., '130/100 mmHg', '60 kg'), qualitative assessments (e.g., 'positive "
        "for adipophilin', 'normal'), or descriptive observations indicating changes or abnormalities "
        "(e.g., 'enlarging mass', 'slowly regained renal function'), directly reflecting the outcome of "
        "a laboratory test or diagnostic evaluation."
    ),
    "detailed_description": (
        "Any explicitly stated modifier or qualifier that provides additional detail about another entity "
        "or event, including descriptors of condition (e.g., 'sudden cardiopulmonary arrest'), extent "
        "(e.g., 'right-sided paralysis'), subtype (e.g., 'group B streptococcus'), or diagnostic type "
        "(e.g., 'serum uric acid'), which are not captured by other entity types."
    ),
}

NCBI_PLAIN_DESCRIPTION = "The disease entity."

# Zero-shot / few-shot semantic phrasing (thesis A.1.2): opens with "The disease
# entities. A disease entity is any ...".
NCBI_SEMANTIC_DESCRIPTION = (
    "The disease entities. A disease entity is any textual mention that (1) corresponds to a specific "
    "disease or class of diseases (e.g., a named syndrome, disorder, or pathological condition) and (2) "
    "is mappable to a distinct biomedical concept (for example via UMLS/MeSH/OMIM) and (3) is relevant "
    "to biomedical information retrieval (i.e., meaningful for physicians or health-care professionals). "
    "Mentions such as pure symptoms, isolated anatomical abnormalities, or general descriptive terms "
    "should not be extracted unless they are part of a recognized disease name."
)

# Fine-tuning semantic phrasing (thesis A.3.2): the parenthetical opens directly with
# "a disease entity is any ..." (no "The disease entities." lead-in).
NCBI_SEMANTIC_DEFINITION = (
    "a disease entity is any textual mention that (1) corresponds to a specific disease or class of "
    "diseases (e.g., a named syndrome, disorder, or pathological condition) and (2) is mappable to a "
    "distinct biomedical concept (for example via UMLS/MeSH/OMIM) and (3) is relevant to biomedical "
    "information retrieval (i.e., meaningful for physicians or health-care professionals). Mentions such "
    "as pure symptoms, isolated anatomical abnormalities, or general descriptive terms should not be "
    "extracted unless they are part of a recognized disease name"
)

# ===============================================================================
# Few-shot examples (thesis Appendix A.2). Order in the prompt: two positive
# examples surrounding one negative (empty) example.
# ===============================================================================

MACCROBAT_FEWSHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "age": [
        {"input": "A 31-year-old man developed diabetes insipidus with urine volume up to 10 to 20 L every 24 hours in 2003.",
         "output": [{"text": "31-year-old", "label": "Age"}]},
        {"input": "Physical examination revealed a blood pressure = 100/60 mm Hg, pulse = 60 beats/min, no jaundice, no stigmata of chronic liver disease, a soft abdomen with mild epigastric tenderness but no rebound tenderness, no abdominal bruit, and no pulsatile abdominal mass.",
         "output": [{"text": "", "label": "Age"}]},
        {"input": "A 60-year-old woman patient was admitted to our hospital on Feb. 18, 2016 because of frequent episodes of hemoptysis for 2 weeks.",
         "output": [{"text": "60-year-old", "label": "Age"}]},
    ],
    "sex": [
        {"input": "A 31-year-old man developed diabetes insipidus with urine volume up to 10 to 20 L every 24 hours in 2003.",
         "output": [{"text": "man", "label": "Sex"}]},
        {"input": "Physical examination revealed a blood pressure = 100/60 mm Hg, pulse = 60 beats/min, no jaundice, no stigmata of chronic liver disease, a soft abdomen with mild epigastric tenderness but no rebound tenderness, no abdominal bruit, and no pulsatile abdominal mass.",
         "output": [{"text": "", "label": "Sex"}]},
        {"input": "A 60-year-old woman patient was admitted to our hospital on Feb. 18, 2016 because of frequent episodes of hemoptysis for 2 weeks.",
         "output": [{"text": "woman", "label": "Sex"}]},
    ],
    "biological_structure": [
        {"input": "Our patient had an esophagogastroduodenoscopy (EGD) and barium swallow that revealed no stricture of her esophagus but failure of primary and secondary peristaltic waves and reflux.",
         "output": [{"text": "esophagus", "label": "Biological structure"}, {"text": "primary and secondary peristaltic waves", "label": "Biological structure"}, {"text": "reflux", "label": "Biological structure"}]},
        {"input": "Unfortunately, after two cycles his shortness of breath worsened, with evidence of further progression on his scans (Fig.1A).",
         "output": [{"text": "", "label": "Biological structure"}]},
        {"input": "A second opinion was requested from the University of Colorado and a computed tomography–guided biopsy of the left upper lobe lesion was performed to permit additional molecular testing.",
         "output": [{"text": "left upper lobe", "label": "Biological structure"}]},
    ],
    "sign_symptom": [
        {"input": "Mild anemia was observed (hemoglobin, 11.3 g/dL), although hemoglobin levels had been 14.6 g/dL prior to the hemoptysis episode.",
         "output": [{"text": "anemia", "label": "Sign symptom"}, {"text": "hemoptysis", "label": "Sign symptom"}]},
        {"input": "Bilateral oophorectomy, total hysterectomy, omentectomy, and sigmoidectomy with regional node dissection were performed (Fig.3a).",
         "output": [{"text": "", "label": "Sign symptom"}]},
        {"input": "A 47-year-old woman presented to the hospital with a 1-month history of abdominal distention.",
         "output": [{"text": "distention", "label": "Sign symptom"}]},
    ],
    "diagnostic_procedure": [
        {"input": "Immunohistochemically, tumor cells from the ovaries and the colon both showed positive expression of cytokeratin 20 (CK20) but no expression of cytokeratin 7 (CK7), confirming that the ovarian tumors were metastases from primary colon cancer (Fig.4a, ​b).",
         "output": [{"text": "Immunohistochemically", "label": "Diagnostic procedure"}, {"text": "cytokeratin 20", "label": "Diagnostic procedure"}, {"text": "CK20", "label": "Diagnostic procedure"}, {"text": "cytokeratin 7", "label": "Diagnostic procedure"}, {"text": "CK7", "label": "Diagnostic procedure"}]},
        {"input": "A right thoracotomy was performed on the fourth intercostal space and bicaval cannulation was established.",
         "output": [{"text": "", "label": "Diagnostic procedure"}]},
        {"input": "His morphological features were: (a) normal mouth opening with missing lower incisors (Fig.1); (b) small chest, as determined by a cardiothoracic examination; and (c) disproportionately short extremities with one additional postaxial digit on each hand (Fig.2).",
         "output": [{"text": "morphological features", "label": "Diagnostic procedure"}, {"text": "cardiothoracic examination", "label": "Diagnostic procedure"}]},
    ],
    "lab_value": [
        {"input": "Pulmonary function tests (PFT) showed a mild restrictive ventilatory defect, with a reduced total lung capacity of 79% (5.94 L), forced vital capacity of 80% (4.18 L) and a forced expiratory volume in one second of 83% (3.72 L).",
         "output": [{"text": "reduced", "label": "Lab value"}, {"text": "79%", "label": "Lab value"}, {"text": "5.94 L", "label": "Lab value"}, {"text": "80%", "label": "Lab value"}, {"text": "4.18 L", "label": "Lab value"}, {"text": "83%", "label": "Lab value"}, {"text": "3.72 L", "label": "Lab value"}]},
        {"input": "The patient underwent a fiberoptic bronchoscopy with bronchoalveolar lavage and transbronchial lung biopsy.",
         "output": [{"text": "", "label": "Lab value"}]},
        {"input": "After two years of follow-up, echocardiography revealed optimal function of the mitral valve and a decrease in systolic pulmonary artery pressure (30 mmHg).",
         "output": [{"text": "optimal function", "label": "Lab value"}, {"text": "decrease", "label": "Lab value"}, {"text": "30 mmHg", "label": "Lab value"}]},
    ],
    "detailed_description": [
        {"input": "He developed multifocal salmon-pink skin discoloration, and swelling and spontaneous pain in the left knee and leg.",
         "output": [{"text": "multifocal", "label": "Detailed description"}, {"text": "spontaneous", "label": "Detailed description"}]},
        {"input": "His serum creatinine level was stabilized at 1.7 mg/dL with maintenance immunosuppressive therapy comprising tacrolimus (3 mg/day), mycophenolate mofetil (1500 mg/day), and prednisone (4 mg every other day).",
         "output": [{"text": "", "label": "Detailed description"}]},
        {"input": "We examined the genomic heat shock protein (HSP) 60 sequence of the blood culture isolate, which resulted in identification of cluster B H. cinaedi.",
         "output": [{"text": "blood culture isolate", "label": "Detailed description"}, {"text": "cluster B", "label": "Detailed description"}]},
    ],
}

NCBI_FEWSHOT_EXAMPLES: List[Dict] = [
    {"input": "Mutations in this gene are responsible for Pendred syndrome and autosomal recessive non-syndromic hearing loss at the DFNB4 locus on chromosome 7q31.",
     "output": [{"text": "Pendred syndrome", "label": "DISEASE"}, {"text": "autosomal recessive non-syndromic hearing loss", "label": "DISEASE"}, {"text": "DFNB4", "label": "DISEASE"}]},
    {"input": "The effects of the mutations at the mRNA and protein level were ascertained by RT-PCR and Western blot analyses.",
     "output": [{"text": "", "label": "DISEASE"}]},
    {"input": "Two missense mutations causing mild hyperphenylalaninemia associated with DNA haplotype 12.",
     "output": [{"text": "hyperphenylalaninemia", "label": "DISEASE"}]},
]


# ===============================================================================
# Prompt builders
# ===============================================================================


def _lead_lower(text: str) -> str:
    """Lower-case only the first character, preserving inline case (mmHg, UMLS)."""
    return text[:1].lower() + text[1:] if text else text


def _format_block(label: str) -> str:
    """The output-format block appended to each request:

    ``[{"text": "extracted value", "label": "<LABEL>"}...]``
    """
    structure = json.dumps({"text": "extracted value", "label": label})
    return "[" + structure + "...]"


def _label_request(description: str, label: str) -> str:
    """Build the entity-specific request from a description and its output label."""
    return "Please extract " + _lead_lower(description) + " Answer must follow the following format:\n" + _format_block(label)


def zeroshot_request(dataset: str, label_key: str = None, semantic: bool = False) -> str:
    """Entity-specific request for the zero-shot / few-shot regimes (no wrapper).

    Args:
        dataset: ``"maccrobat"`` or ``"ncbi"``.
        label_key: for MACCROBAT, one of :data:`MACCROBAT_PROMPT_LABELS` keys.
        semantic: use the semantic definition instead of the plain description.
    """
    if dataset == "ncbi":
        desc = NCBI_SEMANTIC_DESCRIPTION if semantic else NCBI_PLAIN_DESCRIPTION
        return _label_request(desc, NCBI_LABEL)
    if dataset == "maccrobat":
        descriptions = MACCROBAT_SEMANTIC_DESCRIPTIONS if semantic else MACCROBAT_PLAIN_DESCRIPTIONS
        return _label_request(descriptions[label_key], MACCROBAT_PROMPT_LABELS[label_key])
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _finetune_label_list(dataset: str, semantic: bool) -> str:
    """Comma-joined label schema for the fine-tuning request.

    In the semantic variant, the seven defined MACCROBAT labels carry their
    definition in parentheses; the rest appear as bare names.
    """
    if dataset == "ncbi":
        if semantic:
            return "disease (" + NCBI_SEMANTIC_DEFINITION + ")"
        return "disease"
    if dataset == "maccrobat":
        # map output label string -> semantic key, for the seven defined labels
        label_to_key = {v: k for k, v in MACCROBAT_PROMPT_LABELS.items()}
        parts = []
        for label in MACCROBAT_LABELS:
            if semantic and label in label_to_key:
                definition = _lead_lower(MACCROBAT_SEMANTIC_DESCRIPTIONS[label_to_key[label]])
                parts.append(f"{label} ({definition.rstrip('.')})")
            else:
                parts.append(label)
        return ", ".join(parts)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def finetuning_request(dataset: str, semantic: bool = False) -> str:
    """Single-pass, full-schema request for the fine-tuning regime (no wrapper)."""
    label_list = _finetune_label_list(dataset, semantic)
    noun = "entity" if dataset == "ncbi" else "entities"
    example_label = NCBI_LABEL if dataset == "ncbi" else "Age"
    return (
        f"Please extract the following {noun}: {label_list}. "
        "Answer must follow the following format:\n" + _format_block(example_label)
    )


def finetuning_instruction(dataset: str, semantic: bool = False) -> str:
    """Full fine-tuning instruction (wrapper + request), ready to pass to the
    :class:`~cer.core.data.formatter.PromptFormatter` as its ``system_prompt``.

    The formatter appends the sentence as the user turn and the medical text after
    it, yielding ``<wrapper> <request>\\nMedical text:\\n<text>``.
    """
    return FINETUNING_SYSTEM_PROMPT + "\n\n" + finetuning_request(dataset, semantic) + "\n\nMedical text:"


def zeroshot_messages(dataset: str, text: str, label_key: str = None, semantic: bool = False) -> List[Dict]:
    """Chat messages for zero-shot prompting (system + user)."""
    request = zeroshot_request(dataset, label_key, semantic)
    return [
        {"role": "system", "content": ZEROSHOT_FEWSHOT_SYSTEM_PROMPT},
        {"role": "user", "content": request + "\n\nMedical text:\n" + text},
    ]


def fewshot_messages(dataset: str, text: str, label_key: str = None, semantic: bool = False) -> List[Dict]:
    """Chat messages for few-shot prompting (system + example turns + user).

    The examples are identical across the plain and semantic variants; only the
    request phrasing changes.
    """
    request = zeroshot_request(dataset, label_key, semantic)
    examples = NCBI_FEWSHOT_EXAMPLES if dataset == "ncbi" else MACCROBAT_FEWSHOT_EXAMPLES[label_key]
    messages = [{"role": "system", "content": ZEROSHOT_FEWSHOT_SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": request + "\n\nMedical text:\n" + ex["input"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"])})
    messages.append({"role": "user", "content": request + "\n\nMedical text:\n" + text})
    return messages


# ===============================================================================
# CLI — print a prompt so shell scripts can capture the exact thesis wording.
#   python -m cer.prompts finetune maccrobat            # fine-tuning instruction
#   python -m cer.prompts finetune ncbi --semantic
# ===============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a thesis prompt string.")
    parser.add_argument("regime", choices=["finetune"], help="Which prompt to print")
    parser.add_argument("dataset", choices=["maccrobat", "ncbi"], help="Dataset")
    parser.add_argument("--semantic", action="store_true", help="Use semantic label definitions")
    ns = parser.parse_args()
    if ns.regime == "finetune":
        print(finetuning_instruction(ns.dataset, semantic=ns.semantic))
