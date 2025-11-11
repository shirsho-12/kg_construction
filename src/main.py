#!/usr/bin/env python3
"""
Main entry point for KG Construction pipelines.
Supports both text and JSON dataset pipelines with YAML configuration.
"""
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import sys

# Ensure the src directory is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipelines.pipeline import run_pipeline
from pipelines.json_pipeline import run_json_pipeline
from pipelines.schema_refiner import SchemaRefiner
from evaluation.qa_system import QASystem


def setup_logging(config: Dict[str, Any]):
    """Setup logging based on config."""
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    logger = logging.getLogger(__name__)
    logger.info(f"Logging set to level: {log_level}")
    return logger


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert relative paths to absolute paths
    config = resolve_paths(config, config_path.parent)
    return config


def resolve_paths(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """Convert relative paths to absolute paths based on project root."""
    project_root = Path(__file__).resolve().parent.parent

    if isinstance(config, dict):
        for key, value in config.items():
            if key.endswith("_path") or key.endswith("_dir"):
                if isinstance(value, str) and not Path(value).is_absolute():
                    config[key] = str((project_root / value).resolve())
            elif isinstance(value, dict):
                config[key] = resolve_paths(value, config_dir)
    return config


def run_text_pipeline(config: Dict[str, Any], logger):
    """Run text dataset pipeline."""
    logger.info("Running text dataset pipeline...")

    pipeline_config = config.get("pipeline", {})

    run_pipeline(
        data_path=Path(pipeline_config.get("data_path")),
        output_dir=Path(pipeline_config.get("output_dir")),
        use_synonyms=pipeline_config.get("use_synonyms", True),
        compression_method=pipeline_config.get("compression_method", "agglomerative"),
        compression_threshold=pipeline_config.get("compression_threshold", 0.8),
        compress_if_more_than=pipeline_config.get("compress_if_more_than", 30),
    )

    logger.info("Text pipeline completed successfully!")


def run_json_pipeline_main(config: Dict[str, Any], logger):
    """Run JSON dataset pipeline."""
    logger.info("Running JSON dataset pipeline...")

    pipeline_config = config.get("pipeline", {})

    run_json_pipeline(
        data_path=Path(pipeline_config.get("data_path")),
        output_dir=Path(pipeline_config.get("output_dir")),
        use_synonyms=pipeline_config.get("use_synonyms", True),
        compression_method=pipeline_config.get("compression_method", "agglomerative"),
        compression_threshold=pipeline_config.get("compression_threshold", 0.8),
        compress_if_more_than=pipeline_config.get("compress_if_more_than", 30),
        extraction_mode=pipeline_config.get("extraction_mode", "base"),
        chunk_size=pipeline_config.get("chunk_size", 100),
    )

    logger.info("JSON pipeline completed successfully!")


def run_schema_refinement(config: Dict[str, Any], logger):
    """Run schema refinement on existing outputs."""
    logger.info("Running schema refinement...")

    refiner_config = config.get("schema_refiner", {})

    refiner = SchemaRefiner(
        compression_method=refiner_config.get("compression_method", "agglomerative"),
        compression_threshold=refiner_config.get("compression_threshold", 0.8),
        compress_if_more_than=refiner_config.get("compress_if_more_than", 30),
    )

    output_dir = Path(refiner_config.get("output_dir"))
    variant = refiner_config.get("variant", "triplets_only")

    if variant == "triplets_only":
        refiner.refine_schema_triplets_only(
            triplets_file=output_dir
            / refiner_config.get("triplets_file", "triplets.json"),
            output_dir=output_dir / "schema_refinement",
        )
    elif variant == "triplets_text":
        refiner.refine_schema_triplets_text(
            triplets_file=output_dir
            / refiner_config.get("triplets_file", "triplets.json"),
            input_file=output_dir / refiner_config.get("input_file", "results.json"),
            output_dir=output_dir / "schema_refinement",
        )
    elif variant == "triplets_synonyms_text":
        refiner.refine_schema_triplets_synonyms_text(
            triplets_file=output_dir
            / refiner_config.get("triplets_file", "triplets.json"),
            synonyms_file=output_dir
            / refiner_config.get("synonyms_file", "synonyms.json"),
            input_file=output_dir / refiner_config.get("input_file", "results.json"),
            output_dir=output_dir / "schema_refinement",
        )

    logger.info("Schema refinement completed successfully!")


def run_qa_evaluation(config: Dict[str, Any], logger):
    """Run QA evaluation on existing outputs."""
    logger.info("Running QA evaluation...")

    qa_config = config.get("qa_evaluation", {})

    # Load dataset for evaluation
    from datasets import JSONDataset

    dataset = JSONDataset(Path(qa_config.get("data_path")), task_type="qa")

    # Initialize QA system
    qa_system = QASystem()

    # Load knowledge graph
    qa_system.load_knowledge_graph_from_file(Path(qa_config.get("triplets_file")))

    # Evaluate both methods
    for method in ["word_match", "graph_rag"]:
        logger.info(f"Evaluating QA with method: {method}")
        metrics = qa_system.evaluate_qa(dataset, method=method)

        # Save results
        output_file = Path(qa_config.get("output_dir")) / f"qa_results_{method}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            import json

            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"QA evaluation results saved to: {output_file}")
        logger.info(f"Accuracy ({method}): {metrics['accuracy']:.4f}")

    logger.info("QA evaluation completed successfully!")


def create_example_configs():
    """Create example configuration files."""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    file_name = "example"
    # Text pipeline config
    text_config = {
        "mode": "text",
        "pipeline": {
            "data_path": f"data/text/{file_name}.txt",
            "output_dir": f"output/text/{file_name}",
            "use_synonyms": True,
            "compression_method": "agglomerative",
            "compression_threshold": 0.8,
            "compress_if_more_than": 30,
        },
        "logging": {"level": "INFO"},
    }

    with open(configs_dir / "text_pipeline.yaml", "w") as f:
        yaml.dump(text_config, f, default_flow_style=False, indent=2)

    # JSON pipeline config
    json_config = {
        "mode": "json",
        "pipeline": {
            "data_path": f"data/json/{file_name}.json",
            "output_dir": f"output/json/{file_name}",
            "use_synonyms": True,
            "compression_method": "agglomerative",
            "compression_threshold": 0.8,
            "compress_if_more_than": 30,
            "extraction_mode": "base",
            "chunk_size": 100,
        },
        "logging": {"level": "INFO"},
    }

    with open(configs_dir / "json_pipeline.yaml", "w") as f:
        yaml.dump(json_config, f, default_flow_style=False, indent=2)

    # Schema refinement config
    schema_config = {
        "mode": "schema_refiner",
        "schema_refiner": {
            "output_dir": f"output/json/{file_name}",
            "variant": "triplets_synonyms_text",
            "triplets_file": "triplets.json",
            "synonyms_file": "synonyms.json",
            "input_file": "results.json",
            "compression_method": "agglomerative",
            "compression_threshold": 0.8,
            "compress_if_more_than": 30,
        },
        "logging": {"level": "INFO"},
    }

    with open(configs_dir / "schema_refinement.yaml", "w") as f:
        yaml.dump(schema_config, f, default_flow_style=False, indent=2)

    # QA evaluation config
    qa_config = {
        "mode": "qa_evaluation",
        "qa_evaluation": {
            "data_path": f"data/json/{file_name}.json",
            "triplets_file": "output/json/{file_name}/triplets.json",
            "output_dir": "output/json/{file_name}",
        },
        "logging": {"level": "INFO"},
    }

    with open(configs_dir / "qa_evaluation.yaml", "w") as f:
        yaml.dump(qa_config, f, default_flow_style=False, indent=2)

    print(f"Created example configuration files in {configs_dir}/:")
    print("  - text_pipeline.yaml")
    print("  - json_pipeline.yaml")
    print("  - schema_refinement.yaml")
    print("  - qa_evaluation.yaml")
    print("\nEdit these files to match your data paths and preferences.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KG Construction Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run text pipeline
  python src/main.py --config configs/text_pipeline.yaml
  
  # Run JSON pipeline
  python src/main.py --config configs/json_pipeline.yaml
  
  # Run schema refinement
  python src/main.py --config configs/schema_refinement.yaml
  
  # Run QA evaluation
  python src/main.py --config configs/qa_evaluation.yaml
        """,
    )

    parser.add_argument(
        "--config", type=Path, required=False, help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create example configuration files in configs/ directory",
    )

    args = parser.parse_args()

    # Create example configs if requested
    if args.create_configs:
        create_example_configs()
        return

    # Validate required arguments
    if not args.config:
        parser.error("--config is required unless using --create-configs")

    # Load and validate config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        config = load_config(args.config)
        logger = setup_logging(config)

        # Get mode from config
        mode = config.get("mode")
        if not mode:
            parser.error(
                "Config must specify 'mode' field with one of: text, json, "
                "schema_refiner, qa_evaluation"
            )

        if mode not in ["text", "json", "schema_refiner", "qa_evaluation"]:
            parser.error(
                "Invalid mode '{mode}' in config. Must be one of: text, json, "
                "schema_refiner, qa_evaluation"
            )

        logger.info(f"Running pipeline mode: {mode}")

        # Run appropriate pipeline
        if mode == "text":
            run_text_pipeline(config, logger)
        elif mode == "json":
            run_json_pipeline_main(config, logger)
        elif mode == "schema_refiner":
            run_schema_refinement(config, logger)
        elif mode == "qa_evaluation":
            run_qa_evaluation(config, logger)
        else:
            print(f"Error: Unknown mode: {mode}")
            sys.exit(1)

    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
