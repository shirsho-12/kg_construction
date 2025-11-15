import nltk
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from .encoder import Encoder
from utils import logger


class Extractor:
    """
    Extractor class for Open Information Extraction (OIE) using a language model encoder.
    """

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def extract(
        self,
        input_text: str,
        prompt_template: str,
        few_shot_examples: Optional[str] = None,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
        synonyms: bool = False,
    ) -> List[Tuple[str]]:

        filled_prompt = prompt_template.format_map(
            {
                "input_text": input_text,
                "few_shot_examples": few_shot_examples or "",
                "entities_hint": entities_hint or "",
                "relations_hint": relations_hint or "",
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]
        logger.debug("Filled prompt: %s", filled_prompt)
        completion = self.encoder.generate_completion(
            messages, answer_prefix="Triplets: "
        )
        logger.debug("Completion received: %s", completion)
        return self.parse_triplets(completion[0], synonyms=synonyms)

    def extract_with_synonyms(
        self,
        input_text: str,
        prompt_template: str,
        few_shot_examples: Optional[str] = None,
        return_synonyms: bool = True,
    ) -> Union[List[Tuple[str]], Tuple[List[Tuple[str]], defaultdict]]:
        extracted_triplets = self.extract(
            input_text,
            prompt_template,
            few_shot_examples=few_shot_examples,
            synonyms=True,
        )
        logger.debug("Extracted triplets: %s", extracted_triplets)
        grouped_relations = defaultdict(list)
        if len(extracted_triplets) == 0:
            return []
        for triple in extracted_triplets:
            if len(triple) < 3:
                logger.debug(triple)
                continue
            head, tail = triple[0], triple[-1]
            for middle in triple[1:-1]:  # type: ignore
                grouped_relations[(head, tail)].append(middle)
        final_triplets = []
        for (s, o), relations in grouped_relations.items():
            if not relations:
                continue

            best_relation = max(relations, key=self._score_relation)
            final_triplets.append([s, best_relation, o])

        if return_synonyms:
            return final_triplets, grouped_relations
        return final_triplets

    def _score_relation(self, relation: str) -> float:
        # Penalty for length
        relation = relation.replace("_", " ")
        if len(relation.split()) == 0:
            return 0.0
        length_penalty = 1.0 / len(relation.split())

        # Prioritize verbs and expressive words
        pos_tags = nltk.pos_tag(nltk.word_tokenize(relation))
        verb_score = sum(1 for _, tag in pos_tags if tag.startswith("VB"))

        # Combine scores
        score = length_penalty + verb_score
        return score

    def parse_triplets(self, raw_triplets: str, synonyms=False) -> List[Tuple[str]]:
        # Look for enclosing brackets
        u_stack = []
        # Keep spaces for parsing, just normalize newlines
        raw_triplets = raw_triplets.replace("\n", " ")

        collected_triples = []
        for c_idx, c in enumerate(raw_triplets):
            if c == "[":
                u_stack.append(c_idx)
            if c == "]":
                if len(u_stack) == 0:
                    continue
                # NOTE: Assuming no nested brackets
                l = u_stack.pop()
                r = c_idx
                bracketed_str = raw_triplets[
                    l + 1 : r
                ]  # Extract content inside brackets

                # Simple parsing: split by comma and clean quotes
                try:
                    # Split by comma and clean each element
                    elements = []
                    current = ""
                    in_quotes = False
                    quote_char = None

                    for char in bracketed_str:
                        if char in ('"', "'") and (not in_quotes or char == quote_char):
                            in_quotes = not in_quotes
                            if in_quotes:
                                quote_char = char
                            else:
                                quote_char = None
                        elif char == "," and not in_quotes:
                            elements.append(current.strip().strip("\"'"))
                            current = ""
                        else:
                            current += char

                    # Add last element
                    if current:
                        elements.append(current.strip().strip("\"'"))

                    # Clean up empty elements
                    elements = [e for e in elements if e and e.strip()]

                    if len(elements) == 3:
                        if all(e != "" and e != "_" for e in elements):
                            collected_triples.append(elements)
                    elif synonyms and len(elements) > 3:
                        # Handle synonym case: head, rel1, rel2, ..., tail
                        head, tail = elements[0], elements[-1]
                        for middle in elements[1:-1]:
                            collected_triples.append([head, middle, tail])
                            logger.debug(
                                "Collected synonym triplet: %s", [head, middle, tail]
                            )
                    elif len(elements) > 0:
                        # Handle malformed but non-empty case
                        collected_triples.append(elements)

                except Exception as e:
                    logger.error("Error parsing triplet: %s", e)
                    continue
                finally:
                    u_stack.clear()
        logger.debug("Collected triplets: %s", collected_triples)
        return collected_triples

    def __call__(
        self,
        input_text: str,
        prompt_template: str,
        few_shot_examples: Optional[str] = None,
    ) -> Union[List[Tuple[str]], List[List[Tuple[str]]]]:
        logger.debug("Running single extraction for input text.")
        return self.extract(
            input_text, prompt_template, few_shot_examples=few_shot_examples
        )
