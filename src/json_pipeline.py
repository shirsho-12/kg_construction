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
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm
import json

from config import (
    BASE_ENCODER_MODEL,
    EXAMPLE_DATA_PATH_JSON,
    LOGGING_LEVEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)
from datasets import JSONDataset, GraphConstructionEvaluator, CombinedTextDataset
from encoder import Encoder
from oie import OIE
from schema_definer import SchemaDefiner
from qa_system import QASystem
from torch.utils.data import DataLoader
from pipeline_utils import (
    setup_file_logging,
    save_problematic_report,
    add_problematic_case,
    save_synonyms,
)

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def run_json_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "agglomerative",
    compression_threshold: float = 0.8,
    compress_if_more_than: int = 30,
    extraction_mode: str = "base",
    chunk_size: int = 100,
    incremental_schema: bool = False,
    schema_update_frequency: int = 50,
):
    """
    Run complete JSON pipeline for graph construction and QA.

    Args:
        data_path: Path to JSON data file
        output_dir: Output directory for results
        use_synonyms: Whether to use synonym generation
        compression_method: Schema compression method
        compression_threshold: Threshold for compression
        compress_if_more_than: Minimum relations before compression
        extraction_mode: Mode for text extraction - "base", "chunking", or "sentence"
        chunk_size: Number of words per chunk (for chunking mode)
        incremental_schema: Whether to run schema compression incrementally
        schema_update_frequency: Number of relations before running schema update
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

    # Load datasets
    graph_dataset = JSONDataset(data_path, task_type="graph_construction")
    qa_dataset = JSONDataset(data_path, task_type="qa")

    logger.info(f"Loaded {len(graph_dataset)} samples for processing")

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
    
    for chunk_idx, chunk_triplets in enumerate(oie_triplets_list):
        sample_idx = combined_ds.get_sample_index(chunk_idx)
        if chunk_triplets:
            sample_triplets_map[sample_idx].extend(chunk_triplets)
    
    logger.info(
        f"Aggregated triplets: "
        f"{sum(len(t) for t in sample_triplets_map.values())} total triplets "
        f"across {len(graph_dataset)} samples"
    )

    # Process each sample for graph construction
    all_triplets_per_sample: List[List[Tuple[str, str, str]]] = []
    all_results: List[Dict[str, Any]] = []
    
    # Incremental schema tracking
    cumulative_schema = {}
    cumulative_relations = set()
    relation_count = 0
    schema_updates_count = 0

    for i in tqdm(range(len(graph_dataset)), desc="Processing samples"):
        sample = graph_dataset[i]
        entity_context = sample["context"]
        # Use aggregated triplets for this sample
        extracted_triplets = sample_triplets_map.get(i, [])
        
        if not extracted_triplets:
            logger.warning(f"No triplets extracted for sample {i}")
            add_problematic_case(
                problematic_cases,
                text=str(sample.get("id", i)),
                issue="No triplets extracted",
                sample_id=sample.get("id"),
                context=entity_context,
            )

        # Apply schema processing if we have triplets
        if extracted_triplets:
            try:
                # Extract relations from triplets
                sample_relations = set()
                for triplet in extracted_triplets:
                    if isinstance(triplet, (list, tuple)) and len(triplet) >= 2:
                        sample_relations.add(triplet[1])
                    elif isinstance(triplet, str) and "#SEP" in triplet:
                        parts = triplet.split("#SEP")
                        if len(parts) >= 2:
                            sample_relations.add(parts[1])
                
                # Incremental schema: update cumulative relations
                if incremental_schema:
                    cumulative_relations.update(sample_relations)
                    relation_count += len(sample_relations)
                    
                    # Check if we need to run schema update
                    if relation_count >= schema_update_frequency * (schema_updates_count + 1):
                        logger.info(
                            f"Running incremental schema update #{schema_updates_count + 1} "
                            f"({len(cumulative_relations)} unique relations)"
                        )
                        # Generate schema for cumulative relations
                        dummy_text = "Cumulative schema generation"
                        dummy_triplets = [["dummy", rel, "dummy"] for rel in cumulative_relations]
                        
                        schema_list = schema_definer.run(dummy_text, [dummy_triplets])
                        if schema_list and schema_list[0]:
                            cumulative_schema = schema_list[0]
                            
                            # Compress if needed
                            if len(cumulative_schema) > compress_if_more_than:
                                cumulative_schema = schema_definer.compress_schema(
                                    cumulative_schema,
                                    method=compression_method,
                                    threshold=compression_threshold,
                                )
                        schema_updates_count += 1
                
                # Schema definition for current sample
                combined_text = " ".join(entity_context.values())
                
                # Use cumulative schema if incremental, otherwise generate new
                if incremental_schema and cumulative_schema:
                    schema = cumulative_schema
                else:
                    schema_list = schema_definer.run(combined_text, [extracted_triplets])
                    schema = schema_list[0] if schema_list and schema_list[0] else {}

                if schema:
                    # Compression if needed (and not using incremental)
                    if not incremental_schema and len(schema) > compress_if_more_than:
                        compressed_schema = schema_definer.compress_schema(
                            schema,
                            method=compression_method,
                            threshold=compression_threshold,
                        )
                    else:
                        compressed_schema = schema

                    # Swap relations to compressed variants
                    if compressed_schema and compressed_schema != schema:
                        original_to_compressed = {}
                        for orig_rel in sample_relations:
                            # Find best match in compressed schema
                            best_match = orig_rel
                            for comp_rel in compressed_schema.keys():
                                if (
                                    orig_rel.lower() in comp_rel.lower()
                                    or comp_rel.lower() in orig_rel.lower()
                                ):
                                    best_match = comp_rel
                                    break
                            original_to_compressed[orig_rel] = best_match
                        
                        final_triplets = schema_definer.swap_relations_to_compressed(
                            extracted_triplets, original_to_compressed
                        )
                    else:
                        final_triplets = extracted_triplets
                else:
                    final_triplets = extracted_triplets
            except Exception as e:
                logger.error(f"Error in schema processing for sample {i}: {e}")
                final_triplets = extracted_triplets
        else:
            final_triplets = []

        all_triplets_per_sample.append(final_triplets)

        # Store results
        result = {
            "sample_id": sample["id"],
            "type": sample["type"],
            "question": sample["question"],
            "ground_truth_answer": sample["answer"],
            "extracted_triplets": final_triplets,
            "ground_truth_evidences": sample["evidences"],
        }
        all_results.append(result)

    # Log incremental schema stats
    if incremental_schema:
        logger.info(
            f"Incremental schema complete: {schema_updates_count} updates performed, "
            f"{len(cumulative_relations)} unique relations tracked"
        )

    # Save original triplets
    original_output_path = output_dir / "triplets.json"
    schema_definer.save_entities_relations_to_json(
        [t for sample_triplets in all_triplets_per_sample for t in sample_triplets],
        original_output_path,
    )

    # Save compressed triplets (same as original in this pipeline)
    compressed_output_path = output_dir / "triplets_compressed.json"
    schema_definer.save_entities_relations_to_json(
        [t for sample_triplets in all_triplets_per_sample for t in sample_triplets],
        compressed_output_path,
    )

    # Save synonyms with de-duplication
    save_synonyms(synonyms, output_dir / "synonyms.json")

    # Save problematic cases report
    if problematic_cases:
        report_path = output_dir / "problematic_cases.json"
        save_problematic_report(problematic_cases, report_path)
    else:
        logger.info("No problematic cases found.")

    # Evaluate graph construction
    logger.info("Evaluating graph construction performance...")
    graph_metrics = evaluator.evaluate_dataset(graph_dataset, all_triplets_per_sample)

    # Save graph construction results
    graph_results_path = output_dir / "graph_construction_results.json"

    with open(graph_results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": graph_metrics, "samples": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Graph construction results saved to {graph_results_path}")
    logger.info(f"Overall F1: {graph_metrics.get('overall_f1', 0):.3f}")

    # QA Evaluation
    logger.info("Running QA evaluation...")
    qa_system = QASystem(encoder=encoder)

    # For each sample, use its extracted triplets as knowledge graph
    qa_results = []
    for i, sample in enumerate(tqdm(qa_dataset, desc="QA Evaluation")):
        qa_system.load_knowledge_graph(all_triplets_per_sample[i])

        question = sample["question"]
        ground_truth = sample["answer"]

        try:
            qa_result = qa_system.answer_question(question)
            qa_result["sample_id"] = sample["id"]
            qa_result["type"] = sample["type"]
            qa_result["ground_truth_answer"] = ground_truth
            qa_results.append(qa_result)
        except Exception as e:
            logger.error(f"Error in QA for sample {i}: {e}")
            qa_results.append(
                {
                    "sample_id": sample["id"],
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
        incremental_schema=False,
        schema_update_frequency=50,
    )
