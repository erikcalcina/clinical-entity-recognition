import json
import re

# ===============================
# Constants
# ===============================

ENGLISH_SYSTEM_PROMPT = "You are a medical entity extractor from clinical texts. Extract the entities from the text and return them in a structured JSON format."

# =====================================
# Main class
# =====================================


class PromptFormatter:
    def __init__(
        self,
        input_key: str = "text",
        output_key: str = "entities",
        system_prompt: str = ENGLISH_SYSTEM_PROMPT,
        unique_entities: bool = True,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt
        self.unique_entities = unique_entities

    # =====================================
    # Training
    # =====================================

    def format_train_example_batch(self, examples, tokenizer):
        texts = []
        batch_size = len(examples[self.input_key])
        for i in range(batch_size):
            example = {key: examples[key][i] for key in examples.keys()}
            texts.append(self._format_train_example(example, tokenizer))
        return {"text": texts}

    def _format_train_example(self, example, tokenizer):
        entities = [self.format_entities(e) for e in example[self.output_key]]

        if self.unique_entities:
            entities = self.get_unique_entities(entities)

        messages = [
            {"role": "user", "content": f"{self.system_prompt}\n{example[self.input_key]}"},
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps(entities, indent=4, ensure_ascii=False)}\n```",
            },
        ]
        txt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        ).removeprefix("<bos>")

        return txt

    # =====================================
    # Testing
    # =====================================

    def format_test_example(self, example, tokenizer):
        messages = [
            {"role": "user", "content": f"{self.system_prompt}\n{example[self.input_key]}"},
        ]

        txt = tokenizer.apply_chat_template(
            messages,
            # Must add for generation
            add_generation_prompt=True,
            tokenize=False,
        )
        return txt

    # =====================================
    # Entities
    # =====================================

    def extract_entities_from_text(self, text):
        # Try multiple patterns
        patterns = [
            r"```json\s*(\[.*?\]|\{.*?\})\s*```",  # Standard JSON blocks
            r"```\s*(\[.*?\]|\{.*?\})\s*```",  # JSON without explicit 'json'
            r"(\[.*?\]|\{.*?\})",  # Raw JSON anywhere
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                try:
                    output = json.loads(matches[0])
                    if (
                        isinstance(output, list)
                        and all(isinstance(m, dict) for m in output)
                        and all(isinstance(m.get("text"), str) and isinstance(m.get("label"), str) for m in output)
                    ):
                        return output
                    continue
                except json.JSONDecodeError:
                    continue

        return []

    def format_entities(self, entity):
        return {
            "label": entity["label"],
            "text": entity["text"].lower(),
            # removed "start" and "end" keys
        }

    def get_unique_entities(self, entities):
        _entities = set()
        for e in entities:
            _entities.add((e["label"], e["text"]))
        return [{"label": e[0], "text": e[1]} for e in _entities]
