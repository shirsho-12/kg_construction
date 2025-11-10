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
from datasets import TextDataset
from encoder import Encoder
from oie import OIE
from schema_definer import SchemaDefiner
from torch.utils.data import DataLoader
from pipeline_utils import (
    setup_file_logging,
    save_problematic_report,
    add_problematic_case,
    save_synonyms,
    process_oie_results,
)

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "hdbscan",
    compression_threshold: float = 0.6,
    compress_if_more_than: int = 3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir)

    # Track problematic cases
    problematic_cases = []

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

    # Batch OIE extraction without nested DataLoaders
    oie_triplets, synonyms = oie.run(dataloader)
    all_triplets_per_text = process_oie_results(
        oie_triplets, dataset, problematic_cases
    )
    schema_definer.save_entities_relations_to_json(
        all_triplets_per_text, output_dir / "triplets.json"
    )
    
    # Save synonyms with de-duplication
    save_synonyms(synonyms, output_dir / "synonyms.json")

    # Collect all triplets and relations for unified schema generation
    all_triplets: List[Tuple[str, str, str]] = []
    all_relations = set()
    text_triplets_map = []  # Store original triplets per text for later compression

    for text, triplets in tqdm(all_triplets_per_text, desc="Collecting relations"):
        if not triplets:
            text_triplets_map.append((text, []))
            continue

        # Extract relations from triplets for unified schema
        text_relations = []
        for triplet in triplets:
            if isinstance(triplet, str):
                parts = triplet.split("#SEP")
                if len(parts) == 3:
                    all_relations.add(parts[1])
                    text_relations.append(parts[1])
            elif isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                all_relations.add(triplet[1])
                text_relations.append(triplet[1])

        text_triplets_map.append((text, triplets, text_relations))
        all_triplets.extend(triplets)

    # Generate unified schema for all relations
    logger.info("Generating unified schema for %d unique relations", len(all_relations))
    try:
        # Create dummy text with all relations for schema generation
        dummy_text = "Schema generation for all extracted relations"
        dummy_triplets = [["dummy", rel, "dummy"] for rel in all_relations]

        schema_list = schema_definer.run(dummy_text, [dummy_triplets])

        if not schema_list or not schema_list[0]:
            logger.warning("Failed to generate unified schema")
            unified_schema = {}
        else:
            unified_schema = schema_list[0]
            logger.info(
                "Generated unified schema with %d relations", len(unified_schema)
            )

    except Exception as e:
        logger.error("Unified schema generation failed: %s", e)
        unified_schema = {}

    # Compress unified schema if it exceeds threshold
    compressed_schema = unified_schema
    original_to_compressed = {}

    if unified_schema and len(unified_schema) > compress_if_more_than:
        logger.info("Compressing schema from %d relations", len(unified_schema))
        try:
            compressed_schema = schema_definer.compress_schema(
                unified_schema,
                method=compression_method,
                threshold=compression_threshold,
            )

            if compressed_schema:
                logger.info("Compressed to %d relations", len(compressed_schema))
                # Build mapping from original to compressed relations
                for orig_rel in unified_schema.keys():
                    # Find the best compressed match (simple heuristic)
                    best_match = orig_rel  # Default to original
                    for comp_rel in compressed_schema.keys():
                        # Simple matching - could use embedding similarity
                        if (
                            orig_rel.lower() in comp_rel.lower()
                            or comp_rel.lower() in orig_rel.lower()
                        ):
                            best_match = comp_rel
                            break
                    original_to_compressed[orig_rel] = best_match
            else:
                logger.warning("Compression returned empty schema")
                compressed_schema = unified_schema

        except Exception as e:
            logger.error("Schema compression failed: %s", e)
            compressed_schema = unified_schema
    else:
        logger.info(
            "Schema has %d relations (<= %d); skipping compression",
            len(unified_schema),
            compress_if_more_than,
        )

    # Apply compression to all triplets
    final_compressed_triplets = []
    for text, triplets, text_relations in text_triplets_map:
        compressed_triplets = []
        for triplet in triplets:
            if isinstance(triplet, str):
                parts = triplet.split("#SEP")
                if len(parts) == 3:
                    subj, rel, obj = parts
                    new_rel = original_to_compressed.get(rel, rel)
                    compressed_triplets.append((subj, new_rel, obj))
            elif isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                subj, rel, obj = triplet
                new_rel = original_to_compressed.get(rel, rel)
                compressed_triplets.append((subj, new_rel, obj))
        final_compressed_triplets.extend(compressed_triplets)

    # 5) Save original triplets
    original_output_path = output_dir / "triplets.json"
    schema_definer.save_entities_relations_to_json(all_triplets, original_output_path)

    # 6) Save compressed triplets
    compressed_output_path = output_dir / "triplets_compressed.json"
    schema_definer.save_entities_relations_to_json(
        final_compressed_triplets, compressed_output_path
    )

    logger.info(
        "Pipeline complete. Saved %d original triplets to %s",
        len(all_triplets),
        original_output_path,
    )
    logger.info(
        "Saved %d compressed triplets to %s",
        len(final_compressed_triplets),
        compressed_output_path,
    )

    # 7) Save unified schema
    unified_schema_path = output_dir / "schema_definitions.json"
    schema_definer.save_schema_definitions([unified_schema], unified_schema_path)

    # 8) Save compression outcomes
    compression_path = output_dir / "compression_outcomes.json"
    schema_definer.save_compression_outcomes(
        [unified_schema], [compressed_schema], compression_method, compression_path
    )

    # 9) Save problematic cases report
    if problematic_cases:
        report_path = output_dir / "problematic_cases.json"
        save_problematic_report(problematic_cases, report_path)
    else:
        logger.info("No problematic cases found.")


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
