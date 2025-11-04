from typing import Optional, Union
from config import (
    BASE_ENCODER_MODEL,
    LOGGING_LEVEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_TEMPLATE_PATH,
)
from encoder import Encoder
from extractor import Extractor
from torch.utils.data import DataLoader
from logging import Logger
from pathlib import Path

logger = Logger(__name__)
logger.setLevel(LOGGING_LEVEL)


class OIE:
    def __init__(
        self,
        model_name: str = BASE_ENCODER_MODEL,
        prompt_template_file: Optional[Union[str, Path]] = None,
        few_shot_examples_file: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model_name
        self.encoder = Encoder(model_name_or_path=model_name)
        self.extractor = Extractor(
            encoder=self.encoder,
        )
        self.prompt_template = ""
        self.few_shot_examples = ""
        if prompt_template_file:
            self.prompt_template = open(prompt_template_file).read()
        if few_shot_examples_file:
            self.few_shot_examples = open(few_shot_examples_file).read()
        logger.info("OIE initialized with model: %s", model_name)
        logger.debug("Prompt template: %s", self.prompt_template)
        logger.debug("Few-shot examples: %s", self.few_shot_examples)

    def run(self, dataloader: DataLoader):
        oie_triplets = []
        for i, batch in enumerate(dataloader):
            logger.debug(f"Processing batch {i}: {batch}")
            oie_triplet = self.extractor(
                input_text=batch,
                prompt_template=self.prompt_template,
                few_shot_examples=self.few_shot_examples,
            )
            oie_triplets.extend(oie_triplet)
        return oie_triplets


if __name__ == "__main__":
    from dataset import TextDataset
    from torch.utils.data import DataLoader
    from config import EXAMPLE_DATA_PATH_TEXT

    oie = OIE(
        prompt_template_file=OIE_TEMPLATE_PATH,
        few_shot_examples_file=OIE_FEW_SHOT_EXAMPLES_PATH,
    )
    dataset = TextDataset(
        data_path=EXAMPLE_DATA_PATH_TEXT,
        encoder=oie.encoder,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    triplets = oie.run(dataloader=dataloader)
    print(triplets)
    for t in triplets:
        print(t)
