#!/usr/bin/env python3
"""
End-to-end pipeline:
1) OIE extracts entities and relations with synonym generation
2) Best relation selected per subject-object pair
3) Schema definitions generated
4) Schema compressed using clustering
5) Relations swapped to compressed variants
6) Entities and relations saved to JSON
"""
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

from config import (
    BASE_ENCODER_MODEL,
    EXAMPLE_DATA_PATH_TEXT,
    LOGGING_LEVEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)
from dataset import TextDataset
from torch.utils.data import DataLoader
from encoder import Encoder
from oie import OIE
from schema_definer import SchemaDefiner

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "agglomerative",
    compression_threshold: float = 0.8,
    compress_if_more_than: int = 30,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    if use_synonyms:
        oie = OIE(
            encoder=encoder,
            prompt_template_file=OIE_SYNONYMY_PROMPT_PATH,
            few_shot_examples_file=OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
            synonymy=True,
        )
    else:
        oie = OIE(
            encoder=encoder,
            prompt_template_file=OIE_PROMPT_PATH,
            few_shot_examples_file=OIE_FEW_SHOT_EXAMPLES_PATH,
            synonymy=False,
        )
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )

    dataset = TextDataset(data_path=data_path, encoder=encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # Run OIE once on the whole dataset (parallelizable)
    all_triplets_per_text = []

    # Batch OIE extraction without nested DataLoaders
    if use_synonyms:
        oie_triplets, _ = oie.run(dataloader)
        # Flatten results
        for i, text in enumerate(dataset):
            if i < len(oie_triplets) and oie_triplets[i]:
                all_triplets_per_text.append((text, oie_triplets[i]))
            else:
                logger.warning("No triplets extracted for text: %s", text)
                all_triplets_per_text.append((text, []))
    else:
        oie_triplets, _ = oie.run(dataloader)
        for i, text in enumerate(dataset):
            if i < len(oie_triplets) and oie_triplets[i]:
                all_triplets_per_text.append((text, oie_triplets[i]))
            else:
                logger.warning("No triplets extracted for text: %s", text)
                all_triplets_per_text.append((text, []))

    # Process schema generation and compression per text
    all_triplets: List[Tuple[str, str, str]] = []
    for text, triplets in tqdm(all_triplets_per_text, desc="Processing schemas"):
        if not triplets:
            continue

        # Schema definition
        schema_list = schema_definer.run(text, [triplets])
        if not schema_list or not schema_list[0]:
            logger.warning("Empty schema for text: %s", text)
            final_triplets = triplets
        else:
            schema = schema_list[0]
            # Compression only if relations exceed threshold
            if len(schema) > compress_if_more_than:
                compressed_schema = schema_definer.compress_schema(
                    schema,
                    method=compression_method,
                    threshold=compression_threshold,
                )
            else:
                logger.debug(
                    "Schema has %d relations (<= %d); skipping compression.",
                    len(schema),
                    compress_if_more_than,
                )
                compressed_schema = schema
            # Swap relations to compressed variants
            if compressed_schema and compressed_schema != schema:
                original_to_compressed = {}
                for orig, comp in zip(schema.keys(), compressed_schema.keys()):
                    original_to_compressed[orig] = comp
                final_triplets = schema_definer.swap_relations_to_compressed(
                    triplets, original_to_compressed
                )
            else:
                final_triplets = triplets

        all_triplets.extend(final_triplets)

    # 5) Save to JSON
    output_path = output_dir / "triplets.json"
    schema_definer.save_entities_relations_to_json(all_triplets, output_path)
    logger.info(
        "Pipeline complete. Saved %d triplets to %s", len(all_triplets), output_path
    )


if __name__ == "__main__":
    run_pipeline(
        data_path=EXAMPLE_DATA_PATH_TEXT,
        output_dir=Path.cwd()
        / "output"
        / EXAMPLE_DATA_PATH_TEXT.parts[-1].split(".")[0],
        use_synonyms=True,
        compression_method="agglomerative",
        compression_threshold=0.8,
        compress_if_more_than=30,
    )
