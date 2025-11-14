from typing import Union
from pathlib import Path
from triplet_extraction import Encoder
from utils import logger


class SchemaDefiner:
    def __init__(
        self,
        model: Encoder,
        schema_prompt_path: Union[str, Path],
        schema_few_shot_examples_path: Union[str, Path],
        max_rel_length: int = 7,
    ):
        self.model = model
        self.schema_prompt = open(schema_prompt_path).read()
        self.schema_few_shot_examples = open(schema_few_shot_examples_path).read()
        self.max_rel_length = max_rel_length
        logger.debug(
            "SchemaDefiner initialized with schema prompt path: %s", schema_prompt_path
        )

    def run(self, input_text: str, oie_triplets: list):
        logger.debug("Defining schema for input text: %s", input_text)
        relations_list = [triplet[1] for triplet in oie_triplets if len(triplet) == 3]
        filled_prompt = self.schema_prompt.format(
            text=input_text,
            few_shot_examples=self.schema_few_shot_examples,
            relations=relations_list,
            triples=oie_triplets,
        )
        logger.debug("Filled schema prompt: %s", filled_prompt)
        schema = self.model.generate_completion(
            [{"role": "user", "content": filled_prompt}], answer_prefix="OUTPUT::"
        )
        logger.debug("Schema completion received: %s", schema)
        if (
            not schema
            or not schema[0]
            or schema[0].strip() in ["", "\n", "OUTPUT::", "OUTPUT::\n"]
        ):
            logger.warning(
                "Schema generation returned empty; using fallback empty schema."
            )
            return {}
        return self._parse_schema(schema[0])

    def _parse_schema(self, schema_text: str):
        descriptions = schema_text.split("\n")
        schema_dct = {}

        for desc in descriptions:
            if ":" not in desc:
                continue

            colon_idx = desc.index(":")
            relation = desc[:colon_idx].strip()
            definition = desc[colon_idx + 1 :].strip()

            relation = self._strip_numbered_prefix(relation)
            definition = self._strip_numbered_prefix(definition)
            if relation == "{relation}" or relation == "Schema" or not relation:
                continue
            relation = relation.replace('"', "").replace("\\", "")
            definition = definition.replace('"', "").replace("\\", "")
            relation = relation.replace("{", "").replace("}", "")
            definition = definition.replace("{", "").replace("}", "")
            if not definition or definition.strip() in ["", "N/A", "None", "null"]:
                continue

            # Any time the relation is longer than max_rel_length words, ignore
            if len(relation.split()) > self.max_rel_length:
                logger.warning(
                    "Skipping relation longer than %d words: %s",
                    self.max_rel_length,
                    relation,
                )
                continue

            # Only add if both relation and definition are valid after cleaning
            if relation and definition:
                schema_dct[relation] = definition

        logger.info("Parsed schema with %d relations after filtering", len(schema_dct))
        return schema_dct

    def _strip_numbered_prefix(self, text: str) -> str:
        # Remove patterns like '1.', '2.', '1) ', '2) ', etc. from the start
        import re

        text = text.strip()
        text = re.sub(r"^\d+[\.\)]\s*", "", text)
        return text
