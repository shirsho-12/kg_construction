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
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import json

from config import (
    BASE_ENCODER_MODEL,
    EXAMPLE_DATA_PATH_TEXT,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)
from datasets import TextDataset
from triplet_extraction.encoder import Encoder
from triplet_extraction.oie import OIE
from schema_definition import SchemaDefiner, SchemaRefiner, FaissSchemaCompressor
from torch.utils.data import DataLoader
from utils.pipeline_utils import (
    setup_file_logging,
    save_problematic_report,
    save_synonyms,
    process_oie_results,
)

from utils import (
    logger,
    load_triplets_from_file,
    load_synonyms_from_file,
    save_schema_definitions,
)

encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)


def run_oie(use_synonyms: bool, dataloader: DataLoader):
    """
    Run Open Information Extraction on the dataset.

    Args:
        use_synonyms: Whether to use synonym generation
        dataloader: DataLoader containing text data

    Returns:
        Tuple of (oie_triplets, synonyms)
    """
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

    try:
        oie_triplets, synonyms = oie.run(dataloader)
        return oie_triplets, synonyms
    except Exception as e:
        logger.error(f"Error running OIE over dataset: {e}")
        return [], []


def run_schema_definition(input_text: str, oie_triplets: List):
    """
    Run schema definition on the given text and triplets.

    Args:
        input_text: Input text for schema generation
        oie_triplets: List of extracted triplets

    Returns:
        Generated schema dictionary
    """
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )

    try:
        schema = schema_definer.run(input_text, oie_triplets)
        return schema
    except Exception as e:
        logger.error(f"Error in schema definition: {e}")
        return {}


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "faiss_similarity",
    compression_threshold: float = 0.6,
    compress_if_more_than: int = 3,
    run_oie_flag: bool = True,
    run_schema_definition_flag: bool = True,
    run_compression_flag: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, "pipeline_errors.log")

    # Track problematic cases
    problematic_cases: List[Dict[str, Any]] = []

    # Initialize dataset
    dataset = TextDataset(data_path=data_path, encoder=encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize schema definer
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )
    # Run OIE extraction if flag is set
    oie_triplets = []
    synonyms = []
    if run_oie_flag:
        logger.info("Running OIE extraction...")
        oie_triplets, synonyms = run_oie(use_synonyms, dataloader)
        all_triplets_per_text = process_oie_results(
            oie_triplets, dataset, problematic_cases
        )
        with open(output_dir / "triplets.json", "w", encoding="utf-8") as f:
            json.dump(all_triplets_per_text, f, indent=2, ensure_ascii=False)

        # Save synonyms with de-duplication
        if use_synonyms and synonyms:
            save_synonyms(synonyms, output_dir / "synonyms.json")
    else:
        if Path.exists(output_dir / "triplets.json"):
            logger.info(
                f"Loading pre-extracted triplets from {output_dir / 'triplets.json'}"
            )
            all_triplets_per_text = load_triplets_from_file(
                output_dir / "triplets.json"
            )
            if use_synonyms:
                synonyms = load_synonyms_from_file(output_dir / "synonyms.json")
        else:
            logger.error(
                "No pre-extracted triplets found. Please run OIE extraction first."
            )
            return

    # Collect all triplets and relations for unified schema generation
    all_triplets = []
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
    unified_schema = {}
    if run_schema_definition_flag:
        logger.info(
            "Generating unified schema for %d unique relations", len(all_relations)
        )
        try:
            # Create dummy text with all relations for schema generation
            dummy_text = "Schema generation for all extracted relations"
            dummy_triplets = [["dummy", rel, "dummy"] for rel in all_relations]

            unified_schema = run_schema_definition(dummy_text, dummy_triplets)

            if not unified_schema:
                logger.warning("Failed to generate unified schema")
            else:
                logger.info(
                    "Generated unified schema with %d relations", len(unified_schema)
                )
                with open(
                    output_dir / "schema_definitions.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        {"schema": unified_schema}, f, indent=2, ensure_ascii=False
                    )

        except Exception as e:
            logger.error("Unified schema generation failed: %s", e)
            unified_schema = {}
    else:
        if Path.exists(output_dir / "schema_definitions.json"):
            logger.info(
                f"Loading pre-defined schemas from {output_dir / 'schema_definitions.json'}"
            )
            with open(
                output_dir / "schema_definitions.json", "r", encoding="utf-8"
            ) as f:
                schema_data = json.load(f)
                unified_schema = schema_data.get("schema", {})
        else:
            logger.warning(
                "No pre-defined schemas found. Schema definition step will be skipped."
            )

    # Compress unified schema if it exceeds threshold
    compressed_schema = unified_schema
    original_to_compressed = {}

    if (
        run_compression_flag
        # and unified_schema
        # and len(unified_schema) > compress_if_more_than
    ):
        faiss_compressor = FaissSchemaCompressor(encoder=encoder)
        schema_refiner = SchemaRefiner(
            faiss_compressor=faiss_compressor,
            compression_method=compression_method,
            compression_ratio=compression_threshold,
            compress_if_more_than=compress_if_more_than,
        )
        logger.info("Compressing schema from %d relations", len(unified_schema))
        try:
            compressed_schema, compression_map = schema_refiner.refine_schema(
                unified_schema
            )

            if compressed_schema:
                logger.info(f"Compressed to {len(compressed_schema)} relations")

            # Apply compression to triplets for this sample only
            if compressed_schema and compressed_schema != unified_schema:
                final_triplets = schema_refiner.swap_relations_to_compressed(
                    all_triplets, compression_map
                )
            else:
                final_triplets = all_triplets
                logger.warning("Compression returned empty schema")
                compressed_schema = unified_schema

        except Exception as e:
            logger.error("Schema compression failed: %s", e)
            compressed_schema = unified_schema
    else:
        if not run_compression_flag:
            logger.info("Schema compression step skipped.")
        else:
            logger.info(
                "Schema has %d relations (<= %d); skipping compression",
                len(unified_schema),
                compress_if_more_than,
            )

    # 6) Save compressed triplets
    compressed_output_path = output_dir / "triplets_compressed.json"
    with open(compressed_output_path, "w", encoding="utf-8") as f:
        json.dump(final_triplets, f, indent=2, ensure_ascii=False)

    logger.info(
        "Pipeline complete. Saved %d original triplets to %s",
        len(all_triplets),
        output_dir / "triplets.json",
    )
    logger.info(
        "Saved %d compressed triplets to %s",
        len(final_triplets),
        compressed_output_path,
    )

    # 7) Save unified schema
    unified_schema_path = output_dir / "schema_definitions.json"
    save_schema_definitions([{"schema": unified_schema}], unified_schema_path)

    # 8) Save compression outcomes
    compression_path = output_dir / "compression_outcomes.json"
    if run_compression_flag:
        try:
            compression_data = {
                "original_schema": unified_schema,
                "compressed_schema": compressed_schema,
                "compression_method": compression_method,
                "compression_threshold": compression_threshold,
                "original_to_compressed": original_to_compressed,
            }
            with open(compression_path, "w", encoding="utf-8") as f:
                json.dump(compression_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save compression outcomes: {e}")

    # 9) Save problematic cases report
    if problematic_cases:
        report_path = output_dir / "problematic_cases.json"
        save_problematic_report(problematic_cases, report_path)
    else:
        logger.info("No problematic cases found.")


if __name__ == "__main__":
    # Run the pipeline
    run_pipeline(
        data_path=EXAMPLE_DATA_PATH_TEXT,
        output_dir=Path.cwd()
        / "output"
        / EXAMPLE_DATA_PATH_TEXT.parts[-1].split(".")[0],
        use_synonyms=True,
        compression_method="faiss_similarity",
        compression_threshold=0.8,
        compress_if_more_than=30,
        run_oie_flag=True,
        run_schema_definition_flag=True,
        run_compression_flag=True,
    )
