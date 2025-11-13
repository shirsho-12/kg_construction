#!/usr/bin/env python3
"""
Schema Definition Refinement Script

This script allows running schema definition without re-running entity extraction.
It works with both text and JSON pipelines, using previously generated triplet files
from the output directory.

Variants:
1. Triplets only
2. Triplets + input text
3. Triplets + synonyms + input text
"""
from pathlib import Path
from typing import List, Dict, Tuple, Union
import logging
import json
import argparse
from tqdm import tqdm

from core.encoder import Encoder
from core.schema_definer import SchemaDefiner
from config import (
    BASE_ENCODER_MODEL,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
    LOGGING_LEVEL,
)
from datasets import JSONDataset

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class SchemaRefiner:
    """Schema definition refinement without entity extraction."""

    def __init__(
        self,
        compression_method: str = "faiss_similarity",
        compression_threshold: float = 0.8,
        compress_if_more_than: int = 30,
        max_schema_size: int = None,
        compression_ratio: float = None,
    ):
        """
        Initialize schema refiner.

        Args:
            compression_method: Schema compression method
            compression_threshold: Threshold for compression
            compress_if_more_than: Minimum relations before compression
        """
        self.compression_method = compression_method
        self.compression_threshold = compression_threshold
        self.compress_if_more_than = compress_if_more_than
        self.max_schema_size = max_schema_size
        self.compression_ratio = compression_ratio

        # Initialize components
        self.encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
        self.schema_definer = SchemaDefiner(
            model=self.encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        logger.info("Initialized Schema Refiner")

    def load_triplets_from_file(
        self, file_path: Path
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Load triplets from JSON file.

        Args:
            file_path: Path to triplets file

        Returns:
            List of triplet lists per sample
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Triplets file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, dict) and "triplets" in data:
            # Format: {"triplets": [[...], [...]]}
            return data["triplets"]
        elif isinstance(data, dict):
            # Format: {"sample_id": [...], "sample_id2": [...]}
            return list(data.values())
        elif isinstance(data, list):
            # Format: [[...], [...]] or [{"subject": ...,
            # "relation": ..., "object": ...}, ...]
            if all(isinstance(item, list) for item in data):
                return data
            else:
                # Convert list of dicts to list of triplets
                triplets = []
                for item in data:
                    if isinstance(item, dict):
                        triplet = (
                            item.get("subject", ""),
                            item.get("relation", ""),
                            item.get("object", ""),
                        )
                        if all(triplet):  # Only add non-empty triplets
                            triplets.append(triplet)
                return [triplets] if triplets else [[]]
        else:
            raise ValueError(f"Unsupported data format in {file_path}")

    def load_synonyms_from_file(self, file_path: Path) -> List[Dict[str, List[str]]]:
        """
        Load synonyms from JSON file.

        Args:
            file_path: Path to synonyms file

        Returns:
            List of synonym dictionaries per sample
        """
        if not file_path.exists():
            logger.warning(f"Synonyms file not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, dict):
            # Format: {"sample_id": {...}, "sample_id2": {...}}
            return list(data.values())
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported synonyms format in {file_path}")

    def load_input_texts_from_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Load input texts from JSON file using existing JSONDataset.

        Args:
            file_path: Path to file containing input texts

        Returns:
            List of input texts per sample
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Input texts file not found: {file_path}")
            return []

        try:
            # Use existing JSONDataset to handle the file parsing
            dataset = JSONDataset(file_path, task_type="graph_construction")

            # Extract combined texts from the dataset
            texts = []
            for sample in dataset.data:
                if "context" in sample and isinstance(sample["context"], dict):
                    # Combine all context values for this sample
                    combined_text = " ".join(sample["context"].values())
                    texts.append(combined_text)
                elif "context" in sample and isinstance(sample["context"], list):
                    # Context is list of [entity, sentences]
                    sentences = []
                    for entity_sentences in sample["context"]:
                        if (
                            isinstance(entity_sentences, list)
                            and len(entity_sentences) >= 2
                        ):
                            sentences.extend(entity_sentences[1])
                    combined_text = " ".join(sentences)
                    texts.append(combined_text)
                else:
                    # Fallback: try to find any text field
                    for value in sample.values():
                        if isinstance(value, str) and len(value) > 50:
                            texts.append(value)
                            break

            logger.info(f"Loaded {len(texts)} input texts from {file_path}")
            return texts

        except Exception as e:
            logger.error(f"Failed to load input texts using JSONDataset: {e}")
            return []

    def refine_schema_triplets_only(
        self, triplets_file: Path, output_dir: Path
    ) -> None:
        """
        Refine schema using only triplets.

        Args:
            triplets_file: Path to triplets file
            output_dir: Output directory for results
        """
        logger.info("Running schema refinement with triplets only")

        # Load triplets
        triplets_list = self.load_triplets_from_file(triplets_file)
        logger.info(f"Loaded {len(triplets_list)} triplet samples")

        # Process schema for each sample
        schema_results = []
        compressed_schema_results = []

        for i, triplets in enumerate(tqdm(triplets_list, desc="Processing schemas")):
            try:
                # Schema definition
                schema_list = self.schema_definer.run("", [triplets])
                schema = schema_list[0] if schema_list and schema_list[0] else {}
                schema_results.append(schema)

                # Schema compression
                compressed_schema = schema
                if schema and len(schema) > self.compress_if_more_than:
                    logger.info(
                        f"Compressing schema for sample {i} from {len(schema)} relations"
                    )
                    compressed_schema = self.schema_definer.compress_schema(
                        schema,
                        method=self.compression_method,
                        threshold=self.compression_threshold,
                        max_size=self.max_schema_size,
                        compression_ratio=self.compression_ratio,
                    )
                    if compressed_schema:
                        logger.info(f"Compressed to {len(compressed_schema)} relations")

                compressed_schema_results.append(compressed_schema)

            except Exception as e:
                logger.error(f"Error processing schema for sample {i}: {e}")
                schema_results.append({})
                compressed_schema_results.append({})

        # Save results
        self._save_schema_results(
            schema_results, compressed_schema_results, output_dir, "triplets_only"
        )

    def refine_schema_triplets_text(
        self, triplets_file: Path, input_file: Path, output_dir: Path
    ) -> None:
        """
        Refine schema using triplets and input text.

        Args:
            triplets_file: Path to triplets file
            input_file: Path to input file containing texts
            output_dir: Output directory for results
        """
        logger.info("Running schema refinement with triplets + input text")

        # Load data
        triplets_list = self.load_triplets_from_file(triplets_file)
        input_texts = self.load_input_texts_from_file(input_file)

        logger.info(
            f"Loaded {len(triplets_list)} triplet samples and {len(input_texts)} input texts"
        )

        # Ensure we have matching lengths
        min_length = min(len(triplets_list), len(input_texts))
        if len(triplets_list) != len(input_texts):
            logger.warning(
                f"Mismatch in lengths: triplets={len(triplets_list)}, texts={len(input_texts)}. Using {min_length} samples."
            )
            triplets_list = triplets_list[:min_length]
            input_texts = input_texts[:min_length]

        # Process schema for each sample
        schema_results = []
        compressed_schema_results = []

        for i, (triplets, text) in enumerate(
            tqdm(
                zip(triplets_list, input_texts),
                desc="Processing schemas",
                total=min_length,
            )
        ):
            try:
                # Schema definition with text context
                schema_list = self.schema_definer.run(text, [triplets])
                schema = schema_list[0] if schema_list and schema_list[0] else {}
                schema_results.append(schema)

                # Schema compression
                compressed_schema = schema
                if schema and len(schema) > self.compress_if_more_than:
                    logger.info(
                        f"Compressing schema for sample {i} from {len(schema)} relations"
                    )
                    compressed_schema = self.schema_definer.compress_schema(
                        schema,
                        method=self.compression_method,
                        threshold=self.compression_threshold,
                        max_size=self.max_schema_size,
                        compression_ratio=self.compression_ratio,
                    )
                    if compressed_schema:
                        logger.info(f"Compressed to {len(compressed_schema)} relations")

                compressed_schema_results.append(compressed_schema)

            except Exception as e:
                logger.error(f"Error processing schema for sample {i}: {e}")
                schema_results.append({})
                compressed_schema_results.append({})

        # Save results
        self._save_schema_results(
            schema_results, compressed_schema_results, output_dir, "triplets_text"
        )

    def refine_schema_triplets_synonyms_text(
        self,
        triplets_file: Path,
        synonyms_file: Path,
        input_file: Path,
        output_dir: Path,
    ) -> None:
        """
        Refine schema using triplets, synonyms, and input text.

        Args:
            triplets_file: Path to triplets file
            synonyms_file: Path to synonyms file
            input_file: Path to input file containing texts
            output_dir: Output directory for results
        """
        logger.info("Running schema refinement with triplets + synonyms + input text")

        # Load data
        triplets_list = self.load_triplets_from_file(triplets_file)
        synonyms_list = self.load_synonyms_from_file(synonyms_file)
        input_texts = self.load_input_texts_from_file(input_file)

        logger.info(
            f"Loaded {len(triplets_list)} triplet samples, {len(synonyms_list)} synonym samples, and {len(input_texts)} input texts"
        )

        # Ensure we have matching lengths
        min_length = min(len(triplets_list), len(synonyms_list), len(input_texts))
        if len(triplets_list) != len(synonyms_list) or len(triplets_list) != len(
            input_texts
        ):
            logger.warning(
                f"Mismatch in lengths: triplets={len(triplets_list)}, synonyms={len(synonyms_list)}, texts={len(input_texts)}. Using {min_length} samples."
            )
            triplets_list = triplets_list[:min_length]
            synonyms_list = synonyms_list[:min_length]
            input_texts = input_texts[:min_length]

        # Process schema for each sample
        schema_results = []
        compressed_schema_results = []

        for i, (triplets, synonyms, text) in enumerate(
            tqdm(
                zip(triplets_list, synonyms_list, input_texts),
                desc="Processing schemas",
                total=min_length,
            )
        ):
            try:
                # Enhance triplets with synonyms if available
                enhanced_triplets = self._enhance_triplets_with_synonyms(
                    triplets, synonyms
                )

                # Schema definition with text context and enhanced triplets
                schema_list = self.schema_definer.run(text, [enhanced_triplets])
                schema = schema_list[0] if schema_list and schema_list[0] else {}
                schema_results.append(schema)

                # Schema compression
                compressed_schema = schema
                if schema and len(schema) > self.compress_if_more_than:
                    logger.info(
                        f"Compressing schema for sample {i} from {len(schema)} relations"
                    )
                    compressed_schema = self.schema_definer.compress_schema(
                        schema,
                        method=self.compression_method,
                        threshold=self.compression_threshold,
                        max_size=self.max_schema_size,
                        compression_ratio=self.compression_ratio,
                    )
                    if compressed_schema:
                        logger.info(f"Compressed to {len(compressed_schema)} relations")

                compressed_schema_results.append(compressed_schema)

            except Exception as e:
                logger.error(f"Error processing schema for sample {i}: {e}")
                schema_results.append({})
                compressed_schema_results.append({})

        # Save results
        self._save_schema_results(
            schema_results,
            compressed_schema_results,
            output_dir,
            "triplets_synonyms_text",
        )

    def _enhance_triplets_with_synonyms(
        self, triplets: List[Tuple[str, str, str]], synonyms: Dict[str, List[str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Enhance triplets using synonyms.

        Args:
            triplets: Original triplets
            synonyms: Synonym dictionary

        Returns:
            Enhanced triplets list
        """
        if not synonyms:
            return triplets

        enhanced_triplets = list(triplets)  # Copy original triplets

        for key, synonym_list in synonyms.items():
            if "#SEP" in key:
                original_relation = key
                for synonym in synonym_list:
                    # Create new triplets with synonym relations
                    for triplet in triplets:
                        if triplet[1] == original_relation:
                            new_triplet = (triplet[0], synonym, triplet[2])
                            if new_triplet not in enhanced_triplets:
                                enhanced_triplets.append(new_triplet)

        return enhanced_triplets

    def _save_schema_results(
        self,
        schema_results: List[Dict],
        compressed_schema_results: List[Dict],
        output_dir: Path,
        variant: str,
    ) -> None:
        """
        Save schema results to files.

        Args:
            schema_results: List of schema definitions
            compressed_schema_results: List of compressed schema definitions
            output_dir: Output directory
            variant: Variant name for file naming
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save schema definitions
        schema_file = output_dir / f"schema_definitions_{variant}.json"
        self.schema_definer.save_schema_definitions([schema_results], schema_file)

        # Save compressed schema definitions
        compressed_schema_file = output_dir / f"compressed_schemas_{variant}.json"
        self.schema_definer.save_schema_definitions(
            [compressed_schema_results], compressed_schema_file
        )

        # Save combined results
        combined_results = []
        for i, (schema, compressed_schema) in enumerate(
            zip(schema_results, compressed_schema_results)
        ):
            combined_results.append(
                {
                    "sample_id": f"sample_{i}",
                    "schema_definition": schema,
                    "compressed_schema": compressed_schema,
                    "num_relations": len(schema),
                    "num_compressed_relations": len(compressed_schema),
                }
            )

        combined_file = output_dir / f"schema_results_{variant}.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Schema refinement results saved to {output_dir}")
        logger.info(f"Schema definitions: {schema_file}")
        logger.info(f"Compressed schemas: {compressed_schema_file}")
        logger.info(f"Combined results: {combined_file}")


def main():
    """Main function for schema refinement."""
    parser = argparse.ArgumentParser(
        description="Refine schema definitions without entity extraction"
    )

    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing previous results",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["triplets_only", "triplets_text", "triplets_synonyms_text"],
        help="Schema refinement variant",
    )

    # Optional arguments
    parser.add_argument(
        "--triplets-file",
        type=str,
        default="triplets.json",
        help="Name of triplets file (default: triplets.json)",
    )
    parser.add_argument(
        "--synonyms-file",
        type=str,
        default="synonyms.json",
        help="Name of synonyms file (default: synonyms.json)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="results.json",
        help="Name of input file containing texts (default: results.json)",
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default="faiss_similarity",
        choices=["faiss_similarity"],
        help="Schema compression method (default: faiss_similarity)",
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=0.8,
        help="Compression threshold (default: 0.8)",
    )
    parser.add_argument(
        "--compress-if-more-than",
        type=int,
        default=30,
        help="Minimum relations before compression (default: 30)",
    )

    args = parser.parse_args()

    # Check if output directory exists
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

    # Initialize schema refiner
    refiner = SchemaRefiner(
        compression_method=args.compression_method,
        compression_threshold=args.compression_threshold,
        compress_if_more_than=args.compress_if_more_than,
    )

    # Create subdirectory for refinement results
    refinement_dir = args.output_dir / "schema_refinement"
    refinement_dir.mkdir(exist_ok=True)

    # Check required files exist
    triplets_file = args.output_dir / args.triplets_file
    if not triplets_file.exists():
        # Try alternative file names
        alt_names = [
            "triplets_by_id.json",
            "triplets_compressed.json",
            "triplets_compressed_by_id.json",
        ]
        for alt_name in alt_names:
            alt_file = args.output_dir / alt_name
            if alt_file.exists():
                triplets_file = alt_file
                logger.info(f"Using alternative triplets file: {alt_name}")
                break
        else:
            raise FileNotFoundError(f"No triplets file found in {args.output_dir}")

    # Run appropriate variant
    if args.variant == "triplets_only":
        refiner.refine_schema_triplets_only(triplets_file, refinement_dir)

    elif args.variant == "triplets_text":
        input_file = args.output_dir / args.input_file
        if not input_file.exists():
            # Try alternative file names
            alt_names = ["results_by_id.json", "graph_construction_results.json"]
            for alt_name in alt_names:
                alt_file = args.output_dir / alt_name
                if alt_file.exists():
                    input_file = alt_file
                    logger.info(f"Using alternative input file: {alt_name}")
                    break
            else:
                raise FileNotFoundError(f"No input file found in {args.output_dir}")

        refiner.refine_schema_triplets_text(triplets_file, input_file, refinement_dir)

    elif args.variant == "triplets_synonyms_text":
        synonyms_file = args.output_dir / args.synonyms_file
        if not synonyms_file.exists():
            # Try alternative file names
            alt_names = ["synonyms_by_id.json"]
            for alt_name in alt_names:
                alt_file = args.output_dir / alt_name
                if alt_file.exists():
                    synonyms_file = alt_file
                    logger.info(f"Using alternative synonyms file: {alt_name}")
                    break
            else:
                raise FileNotFoundError(f"No synonyms file found in {args.output_dir}")

        input_file = args.output_dir / args.input_file
        if not input_file.exists():
            # Try alternative file names
            alt_names = ["results_by_id.json", "graph_construction_results.json"]
            for alt_name in alt_names:
                alt_file = args.output_dir / alt_name
                if alt_file.exists():
                    input_file = alt_file
                    logger.info(f"Using alternative input file: {alt_name}")
                    break
            else:
                raise FileNotFoundError(f"No input file found in {args.output_dir}")

        refiner.refine_schema_triplets_synonyms_text(
            triplets_file, synonyms_file, input_file, refinement_dir
        )

    logger.info("Schema refinement completed successfully!")


if __name__ == "__main__":
    main()
