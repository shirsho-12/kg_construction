import ast
from typing import List, Optional, Tuple
from encoder import Encoder


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
        completion = self.encoder.generate_completion(
            messages, answer_prefix="Triplets: "
        )
        return self.parse_triplets(completion[0])

    def parse_triplets(self, raw_triplets: str) -> List[Tuple[str]]:
        # Look for enclosing brackets
        u_stack = []
        # remove all '[' and ']' characters
        raw_triplets = (
            raw_triplets.replace("\n", " ")
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
        )

        collected_triples = []
        for c_idx, c in enumerate(raw_triplets):
            if c == "(":
                u_stack.append(c_idx)
            if c == ")":
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
                except Exception as e:
                    pass
        return collected_triples


if __name__ == "__main__":
    from config import BASE_ENCODER_MODEL

    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    extractor = Extractor(encoder=encoder)
    extraction_prompt = (
        "Extract all triplets from the following text. Only output the triples. Do not produce any additional text.\n"
        "Output should be of the form [(entity1, relation, entity2), ...]\n"
        "Text: {input_text}\n"
        "Triplets:"
    )
    extraction = extractor.extract(
        input_text="Paris is the capital of France.",
        prompt_template=extraction_prompt,
    )
    print(extraction)
