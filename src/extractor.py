import ast
import nltk
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from config import LOGGING_LEVEL
from encoder import Encoder
import logging

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class Extractor:
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
                print(triple)
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
        length_penalty = 1.0 / len(relation.split())

        # Prioritize verbs and expressive words
        pos_tags = nltk.pos_tag(nltk.word_tokenize(relation))
        verb_score = sum(1 for _, tag in pos_tags if tag.startswith("VB"))

        # Combine scores
        score = length_penalty + verb_score
        return score

    def batch_extract(
        self,
        input_texts: List[str],
        prompt_template: str,
        few_shot_examples: Optional[str] = None,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[Tuple[str]]]:
        raise NotImplementedError("Batch extraction is not implemented yet.")

    def parse_triplets(self, raw_triplets: str, synonyms=False) -> List[Tuple[str]]:
        # Look for enclosing brackets
        u_stack = []
        raw_triplets = raw_triplets.replace("\n", " ").replace(" ", "")

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
                bracketed_str = raw_triplets[l : r + 1]
                try:
                    parsed_triple = ast.literal_eval(bracketed_str)
                    if len(parsed_triple) == 3 and all(
                        [isinstance(t, str) for t in parsed_triple]
                    ):
                        if all([e != "" and e != "_" for e in parsed_triple]):
                            collected_triples.append(parsed_triple)
                    elif not all(
                        [type(x) == type(parsed_triple[0]) for x in parsed_triple]
                    ):
                        for e_idx, e in enumerate(parsed_triple):
                            if isinstance(e, list):
                                parsed_triple[e_idx] = ", ".join(e)
                        collected_triples.append(parsed_triple)
                    elif (
                        synonyms
                        and len(parsed_triple) > 3
                        and all([isinstance(t, str) for t in parsed_triple])
                    ):
                        head, tail = parsed_triple[0], parsed_triple[-1]
                        for middle in parsed_triple[1:-1]:
                            collected_triples.append([head, middle, tail])
                        logger.debug(
                            "Collected synonym triplet: %s", [head, middle, tail]
                        )
                except Exception as e:
                    logger.error("Error parsing triplet: %s", e)
                    continue
                finally:
                    u_stack.clear()
        logger.debug("Collected triplets: %s", collected_triples)
        return collected_triples

    def __call__(
        self,
        input_text: Union[str, List[str]],
        prompt_template: str,
        few_shot_examples: Optional[str] = None,
    ) -> Union[List[Tuple[str]], List[List[Tuple[str]]]]:
        if isinstance(input_text, list):
            logger.debug("Running batch extraction for input texts.")
            return self.batch_extract(
                input_text, prompt_template, few_shot_examples=few_shot_examples
            )
        logger.debug("Running single extraction for input text.")
        return self.extract(
            input_text, prompt_template, few_shot_examples=few_shot_examples
        )


if __name__ == "__main__":
    from config import BASE_ENCODER_MODEL

    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    extractor = Extractor(encoder=encoder)
    extraction_prompt = (
        "Extract all triplets from the following text. Only output the triples. Do not produce any additional text.\n"
        "Output should be of the form [[entity1, relation, entity2], ...]\n"
        "Text: {input_text}\n"
        "Triplets:"
    )
    extraction = extractor.extract(
        input_text="Paris is the capital of France.",
        prompt_template=extraction_prompt,
    )
    print(extraction)
