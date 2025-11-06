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
from dataset import JSONDataset, GraphConstructionEvaluator
from encoder import Encoder
from oie import OIE
from schema_definer import SchemaDefiner
from qa_system import QASystem

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def run_json_pipeline(
    data_path: Path,
    output_dir: Path,
    use_synonyms: bool = True,
    compression_method: str = "agglomerative",
    compression_threshold: float = 0.8,
    compress_if_more_than: int = 30,
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Process each sample for graph construction
    all_triplets_per_sample = []
    all_results = []
    
    for i in tqdm(range(len(graph_dataset)), desc="Processing samples"):
        sample = graph_dataset[i]
        entity_context = sample["context"]
        
        # Combine all entity sentences into one text for OIE
        combined_text = " ".join(entity_context.values())
        
        # Extract triplets using OIE
        try:
            oie_triplets, _ = oie.run([combined_text])
            if oie_triplets and oie_triplets[0]:
                extracted_triplets = oie_triplets[0]
            else:
                extracted_triplets = []
                logger.warning(f"No triplets extracted for sample {i}")
        except Exception as e:
            logger.error(f"Error extracting triplelets for sample {i}: {e}")
            extracted_triplets = []
        
        # Apply schema processing if we have triplets
        if extracted_triplets:
            try:
                # Schema definition
                schema_list = schema_definer.run(combined_text, [extracted_triplets])
                if schema_list and schema_list[0]:
                    schema = schema_list[0]
                    
                    # Compression if needed
                    if len(schema) > compress_if_more_than:
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
                        for orig, comp in zip(schema.keys(), compressed_schema.keys()):
                            original_to_compressed[orig] = comp
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
            "ground_truth_evidences": sample["evidences"]
        }
        all_results.append(result)
    
    # Evaluate graph construction
    logger.info("Evaluating graph construction performance...")
    graph_metrics = evaluator.evaluate_dataset(graph_dataset, all_triplets_per_sample)
    
    # Save graph construction results
    graph_results_path = output_dir / "graph_construction_results.json"
    import json
    with open(graph_results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": graph_metrics,
            "samples": all_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Graph construction results saved to {graph_results_path}")
    logger.info(f"Overall F1: {graph_metrics.get('overall_f1', 0):.3f}")
    
    # QA Evaluation
    logger.info("Running QA evaluation...")
    qa_system = QASystem(encoder=encoder)
    
    # For each sample, use its extracted triplets as knowledge graph
    qa_results = []
    for i, (sample, triplets) in enumerate(zip(qa_dataset, all_triplets_per_sample)):
        qa_system.load_knowledge_graph(triplets)
        
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
            qa_results.append({
                "sample_id": sample["id"],
                "type": sample["type"],
                "question": question,
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "ground_truth_answer": ground_truth,
                "supporting_triplets": []
            })
    
    # Evaluate QA performance
    qa_metrics = _evaluate_qa_results(qa_results)
    
    # Save QA results
    qa_results_path = output_dir / "qa_results.json"
    with open(qa_results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": qa_metrics,
            "results": qa_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"QA results saved to {qa_results_path}")
    logger.info(f"QA Accuracy: {qa_metrics.get('accuracy', 0):.3f}")
    
    # Save combined report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("JSON Pipeline Evaluation Report\\n")
        f.write("=" * 40 + "\\n\\n")
        
        f.write("Graph Construction Metrics:\\n")
        for key, value in graph_metrics.items():
            f.write(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}\\n")
        
        f.write("\\nQA Metrics:\\n")
        for key, value in qa_metrics.items():
            f.write(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}\\n")
    
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
    high_confidence_accuracy = high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "high_confidence_accuracy": high_confidence_accuracy,
        "total_questions": total,
        "correct_answers": correct,
        "high_confidence_total": high_confidence_total,
        "high_confidence_correct": high_confidence_correct
    }


if __name__ == "__main__":
    # Run the pipeline
    run_json_pipeline(
        data_path=EXAMPLE_DATA_PATH_JSON,
        output_dir=Path.cwd() / "output" / "json_pipeline",
        use_synonyms=True,
        compression_method="agglomerative",
        compression_threshold=0.8,
        compress_if_more_than=30,
    )
