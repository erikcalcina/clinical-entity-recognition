"""Clinical entity recognition — reproducibility code for the master thesis.

Three approaches under one task formulation and one evaluation:

- ``cer.llm``    — LLM LoRA fine-tuning (Unsloth) and prompting.
- ``cer.gliner`` — GLiNER span extractor, fine-tuned.
- ``cer.bert``   — supervised BERT token classification (BIO).

Prompts and training hyperparameters follow the thesis (Appendix A and Section 4.4).
Heavy, optional dependencies (unsloth, gliner, transformers) are imported lazily
inside each approach's modules, so importing :mod:`cer` never requires a GPU stack.
"""

__version__ = "1.0.0"
