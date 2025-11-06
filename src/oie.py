from typing import Optional, Union
from config import (
    BASE_ENCODER_MODEL,
    LOGGING_LEVEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
)
from encoder import Encoder
from extractor import Extractor
from torch.utils.data import DataLoader
from logging import Logger
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class OIE:
    def __init__(
        self,
        encoder: Encoder,
        prompt_template_file: Optional[Union[str, Path]] = None,
        few_shot_examples_file: Optional[Union[str, Path]] = None,
        synonymy: bool = False,
    ):
        self.encoder = encoder
        self.extractor = Extractor(
            encoder=self.encoder,
        )
        self.prompt_template = ""
        self.few_shot_examples = ""
        if prompt_template_file:
            self.prompt_template = open(prompt_template_file).read()
        if few_shot_examples_file:
            self.few_shot_examples = open(few_shot_examples_file).read()
        self.synonymy = synonymy
        logger.info("OIE initialized with model: %s", encoder)
        # logger.debug("Prompt template: %s", self.prompt_template)
        # logger.debug("Few-shot examples: %s", self.few_shot_examples)

    def run(self, dataloader: DataLoader):
        if self.synonymy:
            return self._run_with_synonyms(dataloader)
        else:
            return self._run_base(dataloader)

    def _run_base(self, dataloader: DataLoader):
        oie_triplets = []
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting OIE triplets")):
            logger.debug(f"Processing batch {i}: {batch}")
            for b in batch:
                oie_triplet = self.extractor(
                    input_text=b,
                    prompt_template=self.prompt_template,
                    few_shot_examples=self.few_shot_examples,
                )
                oie_triplets.append(oie_triplet)
        return oie_triplets, None

    def _run_with_synonyms(self, dataloader: DataLoader):
        oie_triplets = []
        oie_synonyms = []
        for i, batch in enumerate(dataloader):
            logger.debug(f"Processing batch {i}: {batch}")
            for b in batch:
                result = self.extractor.extract_with_synonyms(
                    input_text=b,
                    prompt_template=self.prompt_template,
                    few_shot_examples=self.few_shot_examples,
                    return_synonyms=True,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    oie_triplet, oie_synonym = result
                    logger.debug(f"Extracted triplet: {oie_triplet}")
                    logger.debug(f"Extracted synonyms: {oie_synonym}")
                    oie_synonyms.append(oie_synonym)
                    oie_triplets.extend(oie_triplet if isinstance(oie_triplet, list) else [])
                elif isinstance(result, list):
                    # Handle case where only list is returned (no synonyms)
                    logger.debug("Extracted triplets (no synonyms): %s", result)
                    oie_triplets.extend(result)
                    oie_synonyms.append({})
                else:
                    logger.warning("Unexpected result from extract_with_synonyms: %s", result)
                    oie_synonyms.append({})

        return oie_triplets, oie_synonyms


if __name__ == "__main__":
    from dataset import TextDataset
    from torch.utils.data import DataLoader
    from config import EXAMPLE_DATA_PATH_TEXT

    synonymy = True
    if synonymy:
        oie = OIE(
            encoder=Encoder(model_name_or_path=BASE_ENCODER_MODEL),
            prompt_template_file=OIE_SYNONYMY_PROMPT_PATH,
            few_shot_examples_file=OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
            synonymy=True,
        )
    else:
        oie = OIE(
            encoder=Encoder(model_name_or_path=BASE_ENCODER_MODEL),
            prompt_template_file=OIE_PROMPT_PATH,
            few_shot_examples_file=OIE_FEW_SHOT_EXAMPLES_PATH,
            synonymy=False,
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
