#!/usr/bin/env python3
"""
Script to run schema definition and compression pipeline on a JSON dataset.

This script:
1. Loads a JSON dataset using the appropriate Dataset class
2. Extracts triplets and synonyms from the data
3. Runs schema definition on the triplets
4. Applies FAISS compression to the schema
5. Saves outputs from both schema definition and compression steps

Usage:
    python scripts/run_schema_pipeline.py --input data.json --dataset hotpotqa --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    BASE_ENCODER_MODEL,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    OIE_FEW_SHOT_EXAMPLES_PATH,
)

from core.encoder import Encoder
from core.schema_definer import SchemaDefiner
from core.oie import OIE
from datasets.json_dataset import JSONDataset
from datasets.hotpotqa_dataset import HotpotQADataset
from datasets.two_wiki_multihopqa_dataset import TwoWikiMultiHopQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataset_class(dataset_type: str):
    """Get the appropriate dataset class based on type."""
    dataset_classes = {
        "json": JSONDataset,
        "hotpotqa": HotpotQADataset,
        "2wikimultihopqa": TwoWikiMultiHopQADataset,
        "twowikimultihopqa": TwoWikiMultiHopQADataset,
    }

    if dataset_type.lower() not in dataset_classes:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Available: {list(dataset_classes.keys())}"
        )

    return dataset_classes[dataset_type.lower()]


def extract_triplets_and_synonyms(
    dataset: JSONDataset,
    oie: OIE,
    use_synonyms: bool = True,
    max_samples: Optional[int] = None,
) -> tuple[List[List[tuple]], List[str]]:
    """Extract triplets and synonyms from dataset samples."""
    logger.info(f"Extracting triplets from dataset with {len(dataset)} samples")

    all_triplets = []
    all_contexts = []

    # Process samples (limit if specified)
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in range(num_samples):
        sample = dataset[i]
        try:
            logger.debug(f"Processing sample {i}: {sample}")
            context = sample.get("context", "")
            logger.debug(f"Raw context: {context} (type: {type(context)})")

            if isinstance(context, list):
                # Handle context as list of [entity, sentences] pairs (HotpotQA format)
                context_text = " ".join(
                    [
                        " ".join(sentences)
                        if isinstance(sentences, list)
                        else str(sentences)
                        for entity, sentences in context
                    ]
                )
            elif isinstance(context, dict) and context:
                # Handle context as dict of entity: sentences (processed by JSONDataset)
                context_text = " ".join(
                    [f"{entity}: {sentences}" for entity, sentences in context.items()]
                )
            elif isinstance(context, dict) and not context:
                # Empty dict from JSONDataset - try to get original context from raw data
                raw_sample = dataset.data[i]  # Access raw data
                original_context = raw_sample.get("context", "")
                context_text = str(original_context) if original_context else ""
            else:
                # Handle context as simple string
                context_text = str(context)

            logger.debug(f"Processed context text: {context_text}")

            if not context_text.strip():
                logger.warning(f"Empty context for sample {i}, skipping")
                continue

            all_contexts.append(context_text)

            # Extract triplets from context using OIE's extractor
            if use_synonyms:
                result = oie.extractor.extract_with_synonyms(
                    input_text=context_text,
                    prompt_template=oie.prompt_template,
                    few_shot_examples=oie.few_shot_examples,
                    return_synonyms=True,
                )
                if isinstance(result, tuple):
                    triplets, synonyms = result
                else:
                    triplets = result
            else:
                triplets = oie.extractor.extract(
                    input_text=context_text,
                    prompt_template=oie.prompt_template,
                    few_shot_examples=oie.few_shot_examples,
                )

            all_triplets.append(triplets if isinstance(triplets, list) else [])

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{num_samples} samples")

        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            all_triplets.append([])  # Add empty list to maintain alignment
            continue

    logger.info(f"Extracted triplets from {len(all_triplets)} samples")
    return all_triplets, all_contexts


def run_schema_definition(
    schema_definer: SchemaDefiner, triplets_list: List[List[tuple]], contexts: List[str]
) -> Dict[str, str]:
    """Run schema definition on extracted triplets."""
    logger.info("Running schema definition...")

    # Combine all triplets for schema definition
    all_triplets_flat = []
    for triplets in triplets_list:
        all_triplets_flat.extend(triplets)

    if not all_triplets_flat:
        logger.warning("No triplets found for schema definition")
        return {}

    # Combine contexts for schema definition
    combined_context = " ".join(contexts)

    # Run schema definition
    schema = schema_definer.run(combined_context, [all_triplets_flat])

    logger.info(f"Generated schema with {len(schema)} relations")
    return schema


def run_schema_compression(
    schema_definer: SchemaDefiner,
    schema: Dict[str, str],
    compression_method: str = "faiss_max_size",
    max_size: Optional[int] = None,
    compression_ratio: Optional[float] = None,
    similarity_threshold: float = 0.8,
) -> Dict[str, str]:
    """Run FAISS compression on the schema."""
    if not schema:
        logger.warning("Empty schema provided for compression")
        return {}

    logger.info(
        f"Running {compression_method} compression on schema with {len(schema)} relations"
    )

    compressed_schema = schema_definer.compress_schema(
        schema,
        method=compression_method,
        threshold=similarity_threshold,
        max_size=max_size,
        compression_ratio=compression_ratio,
    )

    logger.info(f"Compressed schema to {len(compressed_schema)} relations")
    return compressed_schema


def save_results(
    output_dir: Path,
    original_schema: Dict[str, str],
    compressed_schema: Dict[str, str],
    triplets_list: List[List[tuple]],
    contexts: List[str],
    dataset_info: Dict[str, Any],
):
    """Save all pipeline results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original schema
    schema_file = output_dir / "original_schema.json"
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(original_schema, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved original schema to {schema_file}")

    # Save compressed schema
    compressed_file = output_dir / "compressed_schema.json"
    with open(compressed_file, "w", encoding="utf-8") as f:
        json.dump(compressed_schema, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved compressed schema to {compressed_file}")

    # Save triplets
    triplets_file = output_dir / "extracted_triplets.json"
    # Convert triplets to serializable format
    serializable_triplets = [
        [
            list(triplet) if isinstance(triplet, tuple) else triplet
            for triplet in sample_triplets
        ]
        for sample_triplets in triplets_list
    ]
    with open(triplets_file, "w", encoding="utf-8") as f:
        json.dump(serializable_triplets, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted triplets to {triplets_file}")

    # Save contexts
    contexts_file = output_dir / "contexts.json"
    with open(contexts_file, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved contexts to {contexts_file}")

    # Save dataset info and pipeline metadata
    metadata = {
        "dataset_info": dataset_info,
        "pipeline_results": {
            "original_schema_size": len(original_schema),
            "compressed_schema_size": len(compressed_schema),
            "compression_ratio": (
                len(compressed_schema) / len(original_schema) if original_schema else 0
            ),
            "total_samples_processed": len(contexts),
            "total_triplets_extracted": sum(
                len(triplets) for triplets in triplets_list
            ),
        },
    }

    metadata_file = output_dir / "pipeline_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved pipeline metadata to {metadata_file}")

    # Save summary report
    summary_file = output_dir / "summary_report.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Schema Definition and Compression Pipeline Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset Type: {dataset_info['dataset_type']}\n")
        f.write(f"Input File: {dataset_info['input_file']}\n")
        f.write(f"Samples Processed: {len(contexts)}\n")
        f.write(
            f"Total Triplets Extracted: {sum(len(triplets) for triplets in triplets_list)}\n\n"
        )
        f.write(f"Original Schema Size: {len(original_schema)} relations\n")
        f.write(f"Compressed Schema Size: {len(compressed_schema)} relations\n")
        f.write(
            f"Compression Ratio: {len(compressed_schema) / len(original_schema) if original_schema else 0:.3f}\n\n"
        )

        f.write("Original Schema Relations:\n")
        for i, (rel, defn) in enumerate(original_schema.items(), 1):
            f.write(f"{i:2d}. {rel}: {defn}\n")

        f.write(f"\nCompressed Schema Relations:\n")
        for i, (rel, defn) in enumerate(compressed_schema.items(), 1):
            f.write(f"{i:2d}. {rel}: {defn}\n")

    logger.info(f"Saved summary report to {summary_file}")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run schema definition and compression pipeline on JSON dataset"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=["json", "hotpotqa", "2wikimultihopqa", "twowikimultihopqa"],
        help="Dataset type/class to use",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--compression-method",
        type=str,
        default="faiss_max_size",
        choices=["faiss_max_size", "faiss_ratio", "faiss_similarity"],
        help="Compression method to use (default: faiss_max_size)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=20,
        help="Maximum schema size for faiss_max_size method (default: 20)",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        help="Compression ratio for faiss_ratio method (e.g., 0.5 for 50%)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for faiss_similarity method (default: 0.8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--use-synonyms",
        action="store_true",
        default=True,
        help="Use synonym extraction (default: True)",
    )
    parser.add_argument(
        "--no-synonyms", action="store_true", help="Disable synonym extraction"
    )

    args = parser.parse_args()

    # Handle synonym flags
    use_synonyms = args.use_synonyms and not args.no_synonyms

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output)

    # Validate compression parameters
    if args.compression_method == "faiss_ratio" and args.compression_ratio is None:
        logger.error("--compression-ratio is required when using faiss_ratio method")
        sys.exit(1)

    try:
        logger.info("Initializing pipeline components...")

        # Initialize encoder
        encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)

        # Initialize OIE extractor
        oie = OIE(
            encoder=encoder,
            prompt_template_file=OIE_PROMPT_PATH,
            few_shot_examples_file=OIE_FEW_SHOT_EXAMPLES_PATH,
            synonymy=use_synonyms,
        )

        # Initialize schema definer
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        # Load dataset
        logger.info(f"Loading {args.dataset} dataset from {input_path}")
        dataset_class = get_dataset_class(args.dataset)
        dataset = dataset_class(input_path)

        dataset_info = {
            "dataset_type": args.dataset,
            "input_file": str(input_path),
            "total_samples": len(dataset),
            "samples_processed": min(len(dataset), args.max_samples or len(dataset)),
        }

        # Extract triplets and synonyms
        triplets_list, contexts = extract_triplets_and_synonyms(
            dataset, oie, use_synonyms, args.max_samples
        )

        if not triplets_list or not any(triplets_list):
            logger.warning("No triplets extracted from dataset")
            logger.info("Creating empty results for demonstration purposes")
            
            # Create minimal demo results
            original_schema = {}
            compressed_schema = {}
        else:
            # Run schema definition
            original_schema = run_schema_definition(schema_definer, triplets_list, contexts)
            
            if not original_schema:
                logger.warning("No schema generated from triplets")
                original_schema = {}
                compressed_schema = {}
            else:
                # Run schema compression
                compressed_schema = run_schema_compression(
                    schema_definer,
                    original_schema,
                    compression_method=args.compression_method,
                    max_size=args.max_size if args.compression_method == "faiss_max_size" else None,
                    compression_ratio=args.compression_ratio,
                    similarity_threshold=args.similarity_threshold
                )

        # Save results
        save_results(
            output_dir,
            original_schema,
            compressed_schema,
            triplets_list,
            contexts,
            dataset_info,
        )

        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Original schema: {len(original_schema)} relations")
        logger.info(f"Compressed schema: {len(compressed_schema)} relations")
        logger.info(
            f"Compression ratio: {len(compressed_schema) / len(original_schema) if original_schema else 0:.3f}"
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
