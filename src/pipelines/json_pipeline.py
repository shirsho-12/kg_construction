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
from tqdm import tqdm
import json

from config import (
    BASE_ENCODER_MODEL,
    EXAMPLE_DATA_PATH_JSON,
)
from datasets import (
    JSONDataset,
    GraphConstructionEvaluator,
    CombinedTextDataset,
    TwoWikiMultiHopQADataset,
    HotpotQADataset,
)
from triplet_extraction.encoder import Encoder
from triplet_extraction.oie import OIE
from schema_definition import SchemaDefiner, SchemaRefiner, FaissSchemaCompressor
from evaluation.qa_system import QASystem
from torch.utils.data import DataLoader
from utils.pipeline_utils import (
    setup_file_logging,
    save_problematic_report,
    add_problematic_case,
    evaluate_qa_results,
)

from config import (
    BASE_ENCODER_MODEL,
    OIE_FEW_SHOT_EXAMPLES_PATH,
    OIE_PROMPT_PATH,
    OIE_SYNONYMS_FEW_SHOT_EXAMPLES_PATH,
    OIE_SYNONYMY_PROMPT_PATH,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)

from utils import (
    logger,
    load_triplets_from_file,
    load_synonyms_from_file,
    save_schema_definitions,
)

encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)


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
    compression_method: str = "faiss_similarity",
    compression_threshold: float = 0.8,
    compress_if_more_than: int = 30,
    extraction_mode: str = "base",
    chunk_size: int = 100,
    dataset_type: str = "auto",
    run_oie_flag: bool = False,
    run_schema_definition_flag: bool = False,
    run_compression_flag: bool = True,
    run_qa_flag: bool = False,
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

    # Process each sample independently - no overlap between _id's
    results_by_id: Dict[str, Dict[str, Any]] = {}

    for i in range(len(graph_dataset)):
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
            "compression_mapping": {},
            "synonyms": {},
            "entities": [],
            "relations": [],
        }

    if run_oie_flag:
        logger.info(
            f"Using extraction mode: {extraction_mode}, "
            f"total chunks: {len(combined_ds)}, "
            f"samples: {len(graph_dataset)}"
        )
        oie_triplets, synonyms, sample_triplets_map = run_oie(
            use_synonyms=use_synonyms,
            combined_ds=combined_ds,
            dataloader=dataloader,
            graph_dataset=graph_dataset,
        )
        sample_triplets_map = None
        for i in range(len(graph_dataset)):
            sample = graph_dataset[i]
            entity_context = sample["context"]

            # Get sample _id for isolation
            sample_id = sample.get("_id", f"sample_{i}")
            results_by_id[sample_id]["extracted_triplets"] = oie_triplets[i]
            if oie_triplets[i]:
                results_by_id[sample_id]["entities"] = list(
                    set(
                        [t[0] for t in oie_triplets[i]]
                        + [t[2] for t in oie_triplets[i]]
                    )
                )
                results_by_id[sample_id]["relations"] = list(
                    set([t[1] for t in oie_triplets[i]])
                )
            if use_synonyms:
                # Get all chunk indices for this sample
                chunk_indices = [
                    idx
                    for idx, (s_idx, _) in enumerate(combined_ds.text_chunks)
                    if s_idx == i
                ]
                sample_synonyms = {}
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
            if not results_by_id[sample_id]["extracted_triplets"]:
                logger.warning(f"No triplets found for sample {sample_id}")
                add_problematic_case(
                    problematic_cases,
                    text=str(sample_id),
                    issue="No triplets extracted",
                    sample_id=sample_id,
                    context=entity_context,
                )
    else:
        with open(output_dir / "triplets.json", "r", encoding="utf-8") as f:
            sample_triplets_map = json.load(f)  # key: text id, value: list of triplets
        oie_triplets = load_triplets_from_file(output_dir / "triplets.json")
        if use_synonyms:
            synonyms = load_synonyms_from_file(output_dir / "synonyms.json")

    schema_dct = {}
    if run_schema_definition_flag:
        logger.info("Running schema definition over extracted triplets...")
        for idx, dct in enumerate(tqdm(qa_dataset, desc="Schema Definition")):
            input_text = " ".join(
                dct["context"].values()
            )  # Combine all entity contexts
            if sample_triplets_map:
                oie_triplets_for_sample = sample_triplets_map.get(dct["id"], [])
            else:
                oie_triplets_for_sample = oie_triplets[idx]
            schema = run_schema_definition(input_text, oie_triplets_for_sample)
            schema_dct[dct["id"]] = schema
            results_by_id[dct["id"]]["schema_definition"] = schema
    else:
        with open(output_dir / "schema_definitions.json", "r", encoding="utf-8") as f:
            schema_dct = json.load(f)

    if run_compression_flag:
        logger.info("Running schema compression over defined schemas...")
        faiss_compressor = FaissSchemaCompressor(encoder=encoder)
        schema_refiner = SchemaRefiner(
            faiss_compressor=faiss_compressor,
            compression_method=compression_method,
            compression_ratio=compression_threshold,
            compress_if_more_than=compress_if_more_than,
        )
        for i in range(len(graph_dataset)):
            sample = graph_dataset[i]
            entity_context = sample["context"]

            # Get sample _id for isolation
            sample_id = sample.get("_id", f"sample_{i}")
            extracted_triplets = results_by_id[sample_id]["extracted_triplets"]
            schema = results_by_id[sample_id]["schema_definition"]
            compressed_schema = {}
            try:
                compressed_schema, compression_map = schema_refiner.refine_schema(
                    schema
                )
            except Exception as e:
                logger.error(f"Error in schema compression for sample {sample_id}: {e}")

            if compressed_schema:
                logger.info(f"Compressed to {len(compressed_schema)} relations")

            results_by_id[sample_id]["compressed_schema"] = compressed_schema
            results_by_id[sample_id]["compression_mapping"] = compression_map

            # Apply compression to triplets for this sample only
            if compressed_schema and compressed_schema != schema:
                final_triplets = schema_refiner.swap_relations_to_compressed(
                    extracted_triplets, compression_map
                )
            else:
                final_triplets = extracted_triplets

            results_by_id[sample_id]["compressed_triplets"] = final_triplets

            # # Get synonyms for this sample's chunks only
            # sample_synonyms = {}
            # if use_synonyms and synonyms:
            #     # Get all chunk indices for this sample
            #     chunk_indices = [
            #         idx
            #         for idx, (s_idx, _) in enumerate(combined_ds.text_chunks)
            #         if s_idx == i
            #     ]
            #     for chunk_idx in chunk_indices:
            #         if chunk_idx < len(synonyms) and synonyms[chunk_idx]:
            #             # Convert tuple keys to strings for this sample
            #             for k, v in synonyms[chunk_idx].items():
            #                 key_str = (
            #                     f"{k[0]}#SEP{k[1]}" if isinstance(k, tuple) else str(k)
            #                 )
            #                 if isinstance(v, (list, tuple, set)):
            #                     unique_vals = sorted(set(v))
            #                 else:
            #                     unique_vals = [v]
            #                 sample_synonyms[key_str] = unique_vals

            # results_by_id[sample_id]["synonyms"] = sample_synonyms

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
    save_schema_definitions([schemas_by_id], output_dir / "schema_definitions.json")

    # Save compressed schemas by _id
    compressed_schemas_by_id = {
        sample_id: data["compressed_schema"]
        for sample_id, data in results_by_id.items()
    }
    save_schema_definitions(
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
    if not run_qa_flag:
        return graph_metrics, {}
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
    qa_metrics = evaluate_qa_results(qa_results)

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


def run_oie(use_synonyms, combined_ds, dataloader, graph_dataset):
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

    return oie_triplets_list, synonyms, sample_triplets_map


def run_schema_definition(input_text: str, oie_triplets: list):
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )
    schema = schema_definer.run(input_text, oie_triplets)
    return schema


if __name__ == "__main__":
    # Run the pipeline
    run_json_pipeline(
        data_path=EXAMPLE_DATA_PATH_JSON,
        output_dir=Path.cwd()
        / "output"
        / "qa"
        / EXAMPLE_DATA_PATH_JSON.parts[-1].split(".")[0],
        use_synonyms=True,
        compression_method="faiss_similarity",
        compression_threshold=0.8,
        compress_if_more_than=30,
        extraction_mode="chunking",  # Options: "base", "chunking", "sentence"
        chunk_size=100,
    )
