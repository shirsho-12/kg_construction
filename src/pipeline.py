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


def setup_file_logging(output_dir: Path):
    """Configure file-based error logging with moderate verbosity."""
    log_file = output_dir / "pipeline_errors.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)  # Only warnings and errors
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add to root logger so all modules use it
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.info("Error logging enabled: %s", log_file)


def save_problematic_report(problematic_cases: List[Dict], output_path: Path):
    """Save a detailed report of problematic inputs and their outputs."""
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(problematic_cases, f, indent=2, ensure_ascii=False)
    logger.info(
        "Saved problematic cases report: %s (%d cases)",
        output_path,
        len(problematic_cases),
    )


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
    # Run OIE once on the whole dataset (parallelizable)
    all_triplets_per_text = []

    # Batch OIE extraction without nested DataLoaders
    if use_synonyms:
        oie_triplets, _ = oie.run(dataloader)
        # Flatten results
        texts = [text for text, _ in dataset]
        for i, text in enumerate(texts):
            if i < len(oie_triplets) and oie_triplets[i]:
                # Check for malformed triplets
                malformed = []
                valid_triplets = []
                for triplet in oie_triplets[i]:
                    if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                        malformed.append(str(triplet))
                    else:
                        valid_triplets.append(triplet)

                if malformed:
                    problematic_cases.append(
                        {
                            "input_text": text,
                            "issue": "malformed_triplets",
                            "malformed_triplets": malformed,
                            "valid_triplets": valid_triplets,
                        }
                    )

                all_triplets_per_text.append((text, valid_triplets))
            else:
                logger.warning("No triplets extracted for text: %s", text)
                problematic_cases.append(
                    {
                        "input_text": text,
                        "issue": "no_triplets",
                        "extracted_output": (
                            oie_triplets[i] if i < len(oie_triplets) else None
                        ),
                    }
                )
                all_triplets_per_text.append((text, []))
    else:
        oie_triplets, _ = oie.run(dataloader)
        texts = [text for text, _ in dataset]
        for i, text in enumerate(texts):
            if i < len(oie_triplets) and oie_triplets[i]:
                # Check for malformed triplets
                malformed = []
                valid_triplets = []
                for triplet in oie_triplets[i]:
                    if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                        malformed.append(str(triplet))
                    else:
                        valid_triplets.append(triplet)

                if malformed:
                    problematic_cases.append(
                        {
                            "input_text": text,
                            "issue": "malformed_triplets",
                            "malformed_triplets": malformed,
                            "valid_triplets": valid_triplets,
                        }
                    )

                all_triplets_per_text.append((text, valid_triplets))
            else:
                logger.warning("No triplets extracted for text: %s", text)
                problematic_cases.append(
                    {
                        "input_text": text,
                        "issue": "no_triplets",
                        "extracted_output": (
                            oie_triplets[i] if i < len(oie_triplets) else None
                        ),
                    }
                )
                all_triplets_per_text.append((text, []))

    # Process schema generation and compression per text
    all_triplets: List[Tuple[str, str, str]] = []
    for text, triplets in tqdm(all_triplets_per_text, desc="Processing schemas"):
        if not triplets:
            continue

        try:
            # Schema definition
            schema_list = schema_definer.run(text, [triplets])
        except Exception as e:
            logger.error(
                "Schema definition failed for text: %s. Error: %s", text[:100], e
            )
            problematic_cases.append(
                {
                    "input_text": text,
                    "issue": "schema_definition_failed",
                    "error": str(e),
                    "triplets": triplets,
                }
            )
            all_triplets.extend(triplets)
            continue
        if not schema_list or not schema_list[0]:
            logger.warning("Empty schema for text: %s", text)
            problematic_cases.append(
                {
                    "input_text": text,
                    "issue": "empty_schema",
                    "triplets": triplets,
                    "schema_output": schema_list,
                }
            )
            final_triplets = triplets
        else:
            schema = schema_list[0]
            # Compression only if relations exceed threshold
            try:
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
            except Exception as e:
                logger.error(
                    "Schema compression failed for text: %s. Error: %s", text[:100], e
                )
                problematic_cases.append(
                    {
                        "input_text": text,
                        "issue": "compression_failed",
                        "error": str(e),
                        "schema": schema,
                        "triplets": triplets,
                    }
                )
                final_triplets = triplets

        all_triplets.extend(final_triplets)

    # 5) Save to JSON
    output_path = output_dir / "triplets.json"
    schema_definer.save_entities_relations_to_json(all_triplets, output_path)
    logger.info(
        "Pipeline complete. Saved %d triplets to %s", len(all_triplets), output_path
    )

    # 6) Save problematic cases report
    if problematic_cases:
        report_path = output_dir / "problematic_cases.json"
        save_problematic_report(problematic_cases, report_path)
    else:
        logger.info("No problematic cases found.")


if __name__ == "__main__":
    run_pipeline(
        data_path=EXAMPLE_DATA_PATH_TEXT
        / EXAMPLE_DATA_PATH_TEXT.parts[-1].split(".")[0],
        output_dir=Path.cwd() / "output",
        use_synonyms=True,
        compression_method="agglomerative",
        compression_threshold=0.8,
        compress_if_more_than=30,
    )
