from pathlib import Path
import json
from typing import Any, List, Tuple, Dict, Union
from log import logger
from datasets.json_dataset import JSONDataset


def load_triplets_from_file(file_path: Path) -> List[List[Tuple[str, str, str]]]:
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


def load_synonyms_from_file(file_path: Path) -> List[Dict[str, List[str]]]:
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


def load_input_texts_from_file(file_path: Union[str, Path]) -> List[str]:
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


def save_compression_outcomes(
    original_schemas: List[dict],
    compressed_schemas: List[dict],
    compression_method: str,
    output_path: Union[str, Path],
):
    """Save compression outcomes showing before/after comparison with detailed metrics."""
    import json

    outcomes = []
    total_original = 0
    total_compressed = 0
    successful_compressions = 0

    for i, (orig, comp) in enumerate(zip(original_schemas, compressed_schemas)):
        original_count = len(orig) if orig else 0
        compressed_count = len(comp) if comp else 0
        compression_ratio = (
            compressed_count / original_count if original_count > 0 else 1.0
        )
        reduction_count = original_count - compressed_count
        reduction_percentage = (
            (reduction_count / original_count * 100) if original_count > 0 else 0
        )

        # Check if compression actually occurred
        is_compressed = compressed_count < original_count and original_count > 1
        if is_compressed:
            successful_compressions += 1

        outcome = {
            "input_index": i,
            "original_relations": list(orig.keys()) if orig else [],
            "compressed_relations": list(comp.keys()) if comp else [],
            "original_count": original_count,
            "compressed_count": compressed_count,
            "compression_ratio": compression_ratio,
            "reduction_count": reduction_count,
            "reduction_percentage": round(reduction_percentage, 2),
            "compression_method": compression_method,
            "compression_successful": is_compressed,
            "removed_relations": (
                list(set(orig.keys()) - set(comp.keys())) if orig and comp else []
            ),
        }
        outcomes.append(outcome)

        total_original += original_count
        total_compressed += compressed_count

    # Calculate overall metrics
    overall_compression_ratio = (
        total_compressed / total_original if total_original > 0 else 1.0
    )
    overall_reduction = total_original - total_compressed
    overall_reduction_percentage = (
        (overall_reduction / total_original * 100) if total_original > 0 else 0
    )
    success_rate = (
        (successful_compressions / len(original_schemas) * 100)
        if original_schemas
        else 0
    )

    summary_metrics = {
        "total_inputs": len(original_schemas),
        "total_original_relations": total_original,
        "total_compressed_relations": total_compressed,
        "overall_compression_ratio": round(overall_compression_ratio, 3),
        "overall_reduction": overall_reduction,
        "overall_reduction_percentage": round(overall_reduction_percentage, 2),
        "successful_compressions": successful_compressions,
        "compression_success_rate": round(success_rate, 2),
        "compression_method": compression_method,
    }

    # Save both detailed outcomes and summary metrics
    output_data = {
        "summary_metrics": summary_metrics,
        "detailed_outcomes": outcomes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Compression metrics saved to {output_path}")
    logger.info(
        f"Overall: {total_original} -> {total_compressed} relations "
        f"({overall_reduction} reduced, {overall_reduction_percentage:.1f}% reduction)"
    )
    logger.info(
        f"Success rate: {successful_compressions}/{len(original_schemas)} "
        f"({success_rate:.1f}%)"
    )


def save_entities_relations_to_json(
    oie_triplets: list,
    output_path: Union[str, Path],
):

    results = []
    triplets_text = []  # Also save as text format

    for idx, triplet in enumerate(oie_triplets):
        # Handle different triplet formats
        subj, rel, obj = None, None, None

        if isinstance(triplet, (list, tuple)) and len(triplet) == 1:
            triplet = triplet[0]

        if isinstance(triplet, str):
            new_triplet = triplet.split("#SEP")
            if len(new_triplet) == 3:
                subj, rel, obj = new_triplet
                triplets_text.append(triplet)  # Save original string format
            else:
                logger.warning("Skipping malformed string triplet: %s", triplet)
                continue
        elif (
            isinstance(triplet, (list, tuple))
            and isinstance(triplet[-1], (list, tuple))
            and isinstance(triplet[-1][0], str)
            and "#SEP" in triplet[-1][0]
        ):
            subj, rel, obj = triplet[-1][0].split("#SEP")
            triplets_text.append(triplet[-1][0])  # Save original string format
        elif len(triplet) == 3 and isinstance(triplet, (list, tuple)):
            subj, rel, obj = triplet
            triplets_text.append(f"{subj}#SEP{rel}#SEP{obj}")  # Create string format
        else:
            logger.warning("Skipping malformed triplet: %s", triplet)
            continue

        # Add to results as dictionary format
        if subj and rel and obj:
            results.append({"subject": subj, "relation": rel, "object": obj})

    # Create output data with both formats
    output_data = {
        "triplets": results,
        "triplets_text": triplets_text,
        "count": len(results),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d triples to %s (including text format)", len(results), output_path
    )


def save_schema_definitions(schemas, output_path: Union[str, Path]):
    """Save schema definitions for each input text."""
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d schema definitions to %s", len(schemas), output_path)


def save_schema_results(
    schema_results: List[Dict],
    compressed_schema_results: List[Dict],
    compression_dct: Dict,
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
    save_schema_definitions([schema_results], schema_file)

    # Save compressed schema definitions
    compressed_schema_file = output_dir / f"compressed_schemas_{variant}.json"
    save_schema_definitions([compressed_schema_results], compressed_schema_file)

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

    # save the compression details
    compression_details_file = output_dir / f"compression_details_{variant}.json"
    with open(compression_details_file, "w", encoding="utf-8") as f:
        json.dump(compression_dct, f, indent=2, ensure_ascii=False)

    logger.info(f"Schema refinement results saved to {output_dir}")
    logger.info(f"Schema definitions: {schema_file}")
    logger.info(f"Compressed schemas: {compressed_schema_file}")
    logger.info(f"Combined results: {combined_file}")
