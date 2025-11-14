import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import BASE_ENCODER_MODEL
from triplet_extraction.encoder import Encoder
from triplet_extraction.extractor import Extractor
from datasets import TextDataset
from torch.utils.data import DataLoader
from config import (
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    EXAMPLE_DATA_PATH_TEXT,
)
from triplet_extraction.oie import OIE

# Example usage with transformers
print("Testing transformers encoder...")
encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL, framework="transformers")
text = [{"role": "user", "content": "How are you doing?"}]
embeddings = encoder.encode(text[0]["content"])
print(f"Embeddings shape: {embeddings.shape}")
completion = encoder.generate_completion(text, max_length=50, answer_prefix="Answer: ")
print(completion)


extractor = Extractor(encoder=encoder)
extraction_prompt = """
Extract all triplets from the following text. Only output the triples. Do not produce any additional text.
Output should be of the form [[entity1, relation, entity2], ...]
Text: {input_text}
Triplets:
"""
extraction = extractor.extract(
    input_text="Paris is the capital of France.",
    prompt_template=extraction_prompt,
)
print(extraction)

oie = OIE(
    encoder=encoder,
    prompt_template_file=OIE_SYNONYMY_PROMPT_PATH,
    few_shot_examples_file=OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    synonymy=True,
)
dataset = TextDataset(
    data_path=EXAMPLE_DATA_PATH_TEXT,
    encoder=oie.encoder,
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
triplets, synonyms = oie.run(dataloader=dataloader)
print(triplets)
for t in triplets:
    print(t)
if synonyms:
    print(synonyms)
    for s in synonyms:
        print(s)
