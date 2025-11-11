#!/usr/bin/env python3
"""
End-to-end pipeline for JSON dataset:
1) Load JSON data for graph construction
2) Extract triplets using OIE
3) Evaluate graph construction using evidences
4) Use knowledge graph for QA
5) Evaluate QA performance
"""
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import json

from config_manager import get_config_manager
from datasets import (
    JSONDataset,
    GraphConstructionEvaluator,
    CombinedTextDataset,
    TwoWikiMultiHopQADataset,
    HotpotQADataset,
)
from core.encoder import Encoder
from core.oie import OIE
from core.schema_definer import SchemaDefiner
from evaluation.qa_system import QASystem
from torch.utils.data import DataLoader
from .pipeline_utils import (
    setup_file_logging,
    save_problematic_report,
    add_problematic_case,
)

from config import (
    BASE_ENCODER_MODEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
    LOGGING_LEVEL,
)

# Initialize config manager and get configuration
config_manager = get_config_manager()
base_config = config_manager.base_config

# Setup logging
logging.basicConfig(
    level=getattr(logging, base_config.get("logging", {}).get("level", "INFO").upper()),
    format=base_config.get("logging", {}).get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
)
logger = logging.getLogger(__name__)


def _get_dataset_class(data_path: Path, dataset_type: str):
    """
    Determine the appropriate dataset class based on dataset_type or auto-detection.

    Args:
        data_path: Path to the data file
        dataset_type: Dataset type specifier

    Returns:
        Dataset class to use
    """
    if dataset_type == "auto":
        # Auto-detect based on filename or content
        filename = data_path.name.lower()
        if "2wiki" in filename or "multihop" in filename:
            logger.info("Auto-detected 2WikiMultiHopQA dataset format")
            return TwoWikiMultiHopQADataset
        elif "hotpot" in filename:
            logger.info("Auto-detected HotpotQA dataset format")
            return HotpotQADataset
        else:
            logger.info("Using generic JSONDataset format")
            return JSONDataset
    elif dataset_type == "2wikimultihopqa":
        return TwoWikiMultiHopQADataset
    elif dataset_type == "hotpotqa":
        return HotpotQADataset
    elif dataset_type == "generic":
        return JSONDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_json_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "agglomerative",
    compression_threshold: float = 0.8,
    compress_if_more_than: int = 30,
    extraction_mode: str = "base",
    chunk_size: int = 100,
    dataset_type: str = "auto",
):
    """
    Run complete JSON pipeline for graph construction and QA.

    Each _id is processed independently with complete isolation:
    - Separate schema definition and compression
    - Separate entities, relations, and synonyms
    - No information overlap between different _id's

    Args:
        data_path: Path to JSON data file
        output_dir: Output directory for results
        use_synonyms: Whether to use synonym generation
        compression_method: Schema compression method
        compression_threshold: Threshold for compression
        compress_if_more_than: Minimum relations before compression
        extraction_mode: Mode for text extraction - "base", "chunking", or "sentence"
        chunk_size: Number of words per chunk (for chunking mode)
        dataset_type: Type of dataset - "auto", "2wikimultihopqa", "hotpotqa", or
                      "generic"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir, "json_pipeline_errors.log")

    # Initialize components
    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)

    # Initialize OIE
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

    # Initialize schema definer
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )

    # Initialize evaluator
    evaluator = GraphConstructionEvaluator(encoder=encoder)

    # Determine and load appropriate datasets
    dataset_class = _get_dataset_class(data_path, dataset_type)
    graph_dataset = dataset_class(data_path, task_type="graph_construction")
    qa_dataset = dataset_class(data_path, task_type="qa")

    logger.info(
        f"Loaded {len(graph_dataset)} samples for processing using {dataset_class.__name__}"
    )

    # Track problematic cases
    problematic_cases: List[Dict[str, Any]] = []

    # Build a DataLoader that yields combined text per sample for OIE
    combined_ds = CombinedTextDataset(
        graph_dataset, mode=extraction_mode, chunk_size=chunk_size
    )
    dataloader = DataLoader(combined_ds, batch_size=4, shuffle=False)

    logger.info(
        f"Using extraction mode: {extraction_mode}, "
        f"total chunks: {len(combined_ds)}, "
        f"samples: {len(graph_dataset)}"
    )

    # Extract triplets for all samples using OIE in batch
    try:
        oie_triplets_list, synonyms = oie.run(dataloader)
    except Exception as e:
        logger.error(f"Error running OIE over dataset: {e}")
        oie_triplets_list = [[] for _ in range(len(combined_ds))]
        synonyms = [{} for _ in range(len(combined_ds))]

    # Aggregate chunk triplets back to samples
    logger.info("Aggregating chunk triplets back to samples...")
    sample_triplets_map: Dict[int, List] = {i: [] for i in range(len(graph_dataset))}

    # Since OIE might return flattened results, we need to process chunks manually
    # to maintain proper mapping between chunks and their original samples
    logger.info(f"Processing {len(combined_ds)} chunks individually...")

    chunk_triplets_list = []
    chunk_synonyms_list = []

    # Process each chunk individually to maintain mapping
    for chunk_idx in tqdm(range(len(combined_ds)), desc="Processing chunks"):
        chunk_text = combined_ds[chunk_idx]

        try:
            if use_synonyms:
                result = oie.extractor.extract_with_synonyms(
                    input_text=chunk_text,
                    prompt_template=oie.prompt_template,
                    few_shot_examples=oie.few_shot_examples,
                    return_synonyms=True,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    chunk_triplet, chunk_synonym = result
                    chunk_triplets_list.append(
                        chunk_triplet
                        if isinstance(chunk_triplet, list)
                        else [chunk_triplet]
                    )
                    chunk_synonyms_list.append(chunk_synonym if chunk_synonym else {})
                else:
                    chunk_triplets_list.append(
                        result if isinstance(result, list) else []
                    )
                    chunk_synonyms_list.append({})
            else:
                chunk_triplet = oie.extractor(
                    input_text=chunk_text,
                    prompt_template=oie.prompt_template,
                    few_shot_examples=oie.few_shot_examples,
                )
                chunk_triplets_list.append(
                    chunk_triplet
                    if isinstance(chunk_triplet, list)
                    else [chunk_triplet]
                )
                chunk_synonyms_list.append({})
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            chunk_triplets_list.append([])
            chunk_synonyms_list.append({})

    # Now map chunk results back to samples
    for chunk_idx, chunk_triplets in enumerate(chunk_triplets_list):
        try:
            sample_idx = combined_ds.get_sample_index(chunk_idx)
            if chunk_triplets:
                sample_triplets_map[sample_idx].extend(chunk_triplets)
        except IndexError as e:
            logger.error(f"Index error mapping chunk {chunk_idx}: {e}")
            logger.error(
                f"Chunk results length: {len(chunk_triplets_list)}, Dataset chunks: {len(combined_ds.text_chunks)}"
            )
            continue

    # Replace the original OIE results with our chunk-by-chunk results
    oie_triplets_list = chunk_triplets_list
    synonyms = chunk_synonyms_list

    logger.info(
        f"Aggregated triplets: "
        f"{sum(len(t) for t in sample_triplets_map.values())} total triplets "
        f"across {len(graph_dataset)} samples"
    )

    # Process each sample independently - no overlap between _id's
    results_by_id: Dict[str, Dict[str, Any]] = {}

    for i in tqdm(range(len(graph_dataset)), desc="Processing samples"):
        sample = graph_dataset[i]
        entity_context = sample["context"]

        # Get sample _id for isolation
        sample_id = sample.get("_id", f"sample_{i}")

        # Initialize isolated storage for this _id
        results_by_id[sample_id] = {
            "sample_id": sample_id,
            "type": sample["type"],
            "question": sample["question"],
            "ground_truth_answer": sample["answer"],
            "extracted_triplets": [],
            "compressed_triplets": [],
            "ground_truth_evidences": sample["evidences"],
            "schema_definition": {},
            "compressed_schema": {},
            "synonyms": {},
            "entities": [],
            "relations": [],
        }

        # Use aggregated triplets for this sample
        extracted_triplets = sample_triplets_map.get(i, [])
        results_by_id[sample_id]["extracted_triplets"] = extracted_triplets

        if not extracted_triplets:
            logger.warning(f"No triplets extracted for sample {sample_id}")
            add_problematic_case(
                problematic_cases,
                text=str(sample_id),
                issue="No triplets extracted",
                sample_id=sample_id,
                context=entity_context,
            )
            continue

        # Apply schema processing independently for this _id
        try:
            # Extract entities and relations from triplets for this sample only
            sample_entities = []
            sample_relations = []
            for triplet in extracted_triplets:
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    if triplet[0] not in sample_entities:
                        sample_entities.append(triplet[0])
                    if triplet[1] not in sample_relations:
                        sample_relations.append(triplet[1])
                    if triplet[2] not in sample_entities:
                        sample_entities.append(triplet[2])
                elif isinstance(triplet, str) and "#SEP" in triplet:
                    parts = triplet.split("#SEP")
                    if len(parts) >= 3:
                        if parts[0] not in sample_entities:
                            sample_entities.append(parts[0])
                        if parts[1] not in sample_relations:
                            sample_relations.append(parts[1])
                        if parts[2] not in sample_entities:
                            sample_entities.append(parts[2])

            results_by_id[sample_id]["entities"] = sample_entities
            results_by_id[sample_id]["relations"] = sample_relations

            # Schema definition for this sample only
            combined_text = " ".join(entity_context.values())
            schema_list = schema_definer.run(combined_text, [extracted_triplets])
            schema = schema_list[0] if schema_list and schema_list[0] else {}
            results_by_id[sample_id]["schema_definition"] = schema

            # Compression for this sample only
            compressed_schema = schema
            original_to_compressed = {}

            if schema and len(schema) > compress_if_more_than:
                logger.info(
                    f"Compressing schema for {sample_id} from {len(schema)} relations"
                )
                compressed_schema = schema_definer.compress_schema(
                    schema,
                    method=compression_method,
                    threshold=compression_threshold,
                )

                if compressed_schema:
                    logger.info(f"Compressed to {len(compressed_schema)} relations")
                    # Build mapping for this sample only
                    for orig_rel in sample_relations:
                        best_match = orig_rel
                        for comp_rel in compressed_schema.keys():
                            if (
                                orig_rel.lower() in comp_rel.lower()
                                or comp_rel.lower() in orig_rel.lower()
                            ):
                                best_match = comp_rel
                                break
                        original_to_compressed[orig_rel] = best_match

            results_by_id[sample_id]["compressed_schema"] = compressed_schema

            # Apply compression to triplets for this sample only
            if compressed_schema and compressed_schema != schema:
                final_triplets = schema_definer.swap_relations_to_compressed(
                    extracted_triplets, original_to_compressed
                )
            else:
                final_triplets = extracted_triplets

            results_by_id[sample_id]["compressed_triplets"] = final_triplets

            # Get synonyms for this sample's chunks only
            sample_synonyms = {}
            if use_synonyms and synonyms:
                # Get all chunk indices for this sample
                chunk_indices = [
                    idx
                    for idx, (s_idx, _) in enumerate(combined_ds.text_chunks)
                    if s_idx == i
                ]
                for chunk_idx in chunk_indices:
                    if chunk_idx < len(synonyms) and synonyms[chunk_idx]:
                        # Convert tuple keys to strings for this sample
                        for k, v in synonyms[chunk_idx].items():
                            key_str = (
                                f"{k[0]}#SEP{k[1]}" if isinstance(k, tuple) else str(k)
                            )
                            if isinstance(v, (list, tuple, set)):
                                unique_vals = sorted(set(v))
                            else:
                                unique_vals = [v]
                            sample_synonyms[key_str] = unique_vals

            results_by_id[sample_id]["synonyms"] = sample_synonyms

        except Exception as e:
            logger.error(f"Error in schema processing for sample {sample_id}: {e}")
            results_by_id[sample_id]["compressed_triplets"] = extracted_triplets

    # Save results by _id - complete isolation
    logger.info("Saving isolated results by _id...")

    # Save all results keyed by _id
    results_output_path = output_dir / "results.json"

    with open(results_output_path, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for sample_id, data in results_by_id.items():
            serializable_results[sample_id] = {
                k: v if not isinstance(v, set) else list(v) for k, v in data.items()
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    # Save original triplets by _id
    original_triplets_by_id = {
        sample_id: data["extracted_triplets"]
        for sample_id, data in results_by_id.items()
    }
    original_output_path = output_dir / "triplets.json"
    with open(original_output_path, "w", encoding="utf-8") as f:
        json.dump(original_triplets_by_id, f, indent=2, ensure_ascii=False)

    # Save compressed triplets by _id
    compressed_triplets_by_id = {
        sample_id: data["compressed_triplets"]
        for sample_id, data in results_by_id.items()
    }
    compressed_output_path = output_dir / "triplets_compressed.json"
    with open(compressed_output_path, "w", encoding="utf-8") as f:
        json.dump(compressed_triplets_by_id, f, indent=2, ensure_ascii=False)

    # Save schema definitions by _id
    schemas_by_id = {
        sample_id: data["schema_definition"]
        for sample_id, data in results_by_id.items()
    }
    schema_definer.save_schema_definitions(
        [schemas_by_id], output_dir / "schema_definitions.json"
    )

    # Save compressed schemas by _id
    compressed_schemas_by_id = {
        sample_id: data["compressed_schema"]
        for sample_id, data in results_by_id.items()
    }
    schema_definer.save_schema_definitions(
        [compressed_schemas_by_id], output_dir / "compressed_schemas.json"
    )

    # Save synonyms by _id
    synonyms_by_id = {
        sample_id: data["synonyms"] for sample_id, data in results_by_id.items()
    }
    with open(output_dir / "synonyms.json", "w", encoding="utf-8") as f:
        json.dump(synonyms_by_id, f, indent=2, ensure_ascii=False)

    # Save entities and relations by _id
    entities_by_id = {
        sample_id: data["entities"] for sample_id, data in results_by_id.items()
    }
    with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump(entities_by_id, f, indent=2, ensure_ascii=False)

    relations_by_id = {
        sample_id: data["relations"] for sample_id, data in results_by_id.items()
    }
    with open(output_dir / "relations.json", "w", encoding="utf-8") as f:
        json.dump(relations_by_id, f, indent=2, ensure_ascii=False)

    # Save problematic cases report
    if problematic_cases:
        report_path = output_dir / "problematic_cases.json"
        save_problematic_report(problematic_cases, report_path)
    else:
        logger.info("No problematic cases found.")

    # Evaluate graph construction using isolated results
    logger.info("Evaluating graph construction performance...")
    evaluator = GraphConstructionEvaluator()

    # Extract triplets list for evaluation (maintain order)
    all_triplets_per_sample = []
    for i in range(len(graph_dataset)):
        sample = graph_dataset[i]
        sample_id = sample.get("_id", f"sample_{i}")
        all_triplets_per_sample.append(results_by_id[sample_id]["compressed_triplets"])

    graph_metrics = evaluator.evaluate_dataset(graph_dataset, all_triplets_per_sample)

    # Save graph construction results
    graph_results_path = output_dir / "graph_construction_results.json"

    with open(graph_results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": graph_metrics, "samples_by_id": results_by_id},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Graph construction results saved to {graph_results_path}")
    logger.info(f"Overall F1: {graph_metrics.get('overall_f1', 0):.3f}")

    # QA Evaluation
    logger.info("Running QA evaluation...")
    qa_system = QASystem(encoder=encoder)

    # For each sample, use its extracted triplets as knowledge graph (isolated by _id)
    qa_results = []
    for i, sample in enumerate(tqdm(qa_dataset, desc="QA Evaluation")):
        sample_id = sample.get("_id", f"sample_{i}")
        # Use this sample's isolated triplets
        qa_system.load_knowledge_graph(results_by_id[sample_id]["compressed_triplets"])

        question = sample["question"]
        ground_truth = sample["answer"]

        try:
            qa_result = qa_system.answer_question(question)
            qa_result["sample_id"] = sample_id  # Use _id instead of id
            qa_result["type"] = sample["type"]
            qa_result["ground_truth_answer"] = ground_truth
            qa_results.append(qa_result)
        except Exception as e:
            logger.error(f"Error in QA for sample {sample_id}: {e}")
            qa_results.append(
                {
                    "sample_id": sample_id,  # Use _id instead of id
                    "type": sample["type"],
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "ground_truth_answer": ground_truth,
                    "supporting_triplets": [],
                }
            )

    # Evaluate QA performance
    qa_metrics = _evaluate_qa_results(qa_results)

    # Save QA results
    qa_results_path = output_dir / "qa_results.json"
    with open(qa_results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": qa_metrics, "results": qa_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"QA results saved to {qa_results_path}")
    logger.info(f"QA Accuracy: {qa_metrics.get('accuracy', 0):.3f}")

    # Save combined report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("JSON Pipeline Evaluation Report\\n")
        f.write("=" * 40 + "\\n\\n")

        f.write("Graph Construction Metrics:\\n")
        for key, value in graph_metrics.items():
            f.write(
                f"  {key}: {value:.3f}"
                if isinstance(value, float)
                else f"  {key}: {value}\\n"
            )

        f.write("\\nQA Metrics:\\n")
        for key, value in qa_metrics.items():
            f.write(
                f"  {key}: {value:.3f}"
                if isinstance(value, float)
                else f"  {key}: {value}\\n"
            )

    logger.info(f"Evaluation complete. Report saved to {report_path}")

    return graph_metrics, qa_metrics


def _evaluate_qa_results(qa_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate QA results."""
    correct = 0
    total = 0
    high_confidence_correct = 0
    high_confidence_total = 0

    for result in qa_results:
        predicted = result.get("answer", "").strip().lower()
        ground_truth = result.get("ground_truth_answer", "").strip().lower()
        confidence = result.get("confidence", 0.0)

        if ground_truth and predicted:
            total += 1
            # Simple matching - can be enhanced
            if ground_truth in predicted or predicted in ground_truth:
                correct += 1

            if confidence > 0.7:
                high_confidence_total += 1
                if ground_truth in predicted or predicted in ground_truth:
                    high_confidence_correct += 1

    accuracy = correct / total if total > 0 else 0.0
    high_confidence_accuracy = (
        high_confidence_correct / high_confidence_total
        if high_confidence_total > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "high_confidence_accuracy": high_confidence_accuracy,
        "total_questions": total,
        "correct_answers": correct,
        "high_confidence_total": high_confidence_total,
        "high_confidence_correct": high_confidence_correct,
    }


if __name__ == "__main__":
    # Run the pipeline
    run_json_pipeline(
        data_path=EXAMPLE_DATA_PATH_JSON,
        output_dir=Path.cwd()
        / "output"
        / "json"
        / EXAMPLE_DATA_PATH_JSON.parts[-1].split(".")[0],
        use_synonyms=True,
        compression_method="agglomerative",
        compression_threshold=0.8,
        compress_if_more_than=30,
        extraction_mode="chunking",  # Options: "base", "chunking", "sentence"
        chunk_size=100,
    )
