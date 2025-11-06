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
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    all_triplets: List[Tuple[str, str, str]] = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        logger.debug(f"Processing batch {batch_idx}")
        for text in batch:
            # 1) OIE extraction with synonyms
            if use_synonyms:
                triplets, _ = oie.run(DataLoader([text], batch_size=1))
                triplets = triplets[0] if triplets else []
            else:
                triplets, _ = oie.run(DataLoader([text], batch_size=1))
                triplets = triplets[0] if triplets else []

            if not triplets:
                logger.warning("No triplets extracted for text: %s", text)
                continue

            # 2) Schema definition
            schema_list = schema_definer.run(text, [triplets])
            if not schema_list or not schema_list[0]:
                logger.warning("Empty schema for text: %s", text)
                # Use raw relations without compression
                final_triplets = triplets
            else:
                schema = schema_list[0]
                # 3) Compression
                compressed_schema = schema_definer.compress_schema(
                    schema, method=compression_method, threshold=compression_threshold
                )
                # 4) Swap relations to compressed variants
                if compressed_schema:
                    # Create mapping from original to compressed relations
                    original_to_compressed = {}
                    # Simple heuristic: map by order; can be improved by clustering metadata
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
        output_dir=Path.cwd() / "output",
        use_synonyms=True,
        compression_method="agglomerative",
        compression_threshold=0.8,
    )
