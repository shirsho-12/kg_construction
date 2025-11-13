#!/usr/bin/env python3
"""
Example usage of the schema pipeline script.

This example shows how to run schema definition and compression on different datasets.
"""

import subprocess
import sys
from pathlib import Path

# Get the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


def run_pipeline_example():
    """Run example pipeline commands."""

    print("=== Schema Pipeline Examples ===\n")

    # Example 1: HotpotQA dataset with max size compression
    print("Example 1: HotpotQA with max size compression (20 relations)")
    cmd1 = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_schema_pipeline.py"),
        "--input",
        "data/json/hotpotqa.json",
        "--dataset",
        "hotpotqa",
        "--output",
        "output/hotpotqa_schema_pipeline",
        "--compression-method",
        "faiss_max_size",
        "--max-size",
        "20",
    ]
    print(f"Command: {' '.join(cmd1)}")
    print()

    # Example 2: 2WikiMultiHopQA with ratio compression
    print("Example 2: 2WikiMultiHopQA with ratio compression (60%)")
    cmd2 = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_schema_pipeline.py"),
        "--input",
        "data/json/2wikimultihopqa.json",
        "--dataset",
        "2wikimultihopqa",
        "--output",
        "output/2wikimultihopqa_schema_pipeline",
        "--compression-method",
        "faiss_ratio",
        "--compression-ratio",
        "0.6",
    ]
    print(f"Command: {' '.join(cmd2)}")
    print()

    # Example 3: Generic JSON with similarity compression
    print("Example 3: Generic JSON with similarity compression")
    cmd3 = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_schema_pipeline.py"),
        "--input",
        "data/json/test.json",
        "--dataset",
        "json",
        "--output",
        "output/json_schema_pipeline",
        "--compression-method",
        "faiss_similarity",
        "--similarity-threshold",
        "0.7",
        "--no-synonyms",  # Disable synonyms for faster processing
    ]
    print(f"Command: {' '.join(cmd3)}")
    print()

    print("=== Pipeline Output Structure ===")
    print("Each pipeline run creates the following files:")
    print("  - original_schema.json       # Schema before compression")
    print("  - compressed_schema.json     # Schema after compression")
    print("  - extracted_triplets.json    # All extracted triplets")
    print("  - contexts.json              # Original text contexts")
    print("  - pipeline_metadata.json     # Processing statistics")
    print("  - summary_report.txt         # Human-readable summary")
    print()

    print("=== Usage Tips ===")
    print("1. Use --max-samples for testing with large datasets")
    print("2. Choose compression method based on your needs:")
    print("   - faiss_max_size: Exact number of relations")
    print("   - faiss_ratio: Percentage of original size")
    print("   - faiss_similarity: Group by semantic similarity")
    print("3. Use --no-synonyms to speed up processing")
    print("4. Check pipeline_metadata.json for processing statistics")


def create_sample_data():
    """Create a small sample JSON file for testing."""
    sample_data = [
        {
            "id": "sample_1",
            "question": "Where was Albert Einstein born?",
            "context": "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
            "answer": "Ulm, Germany",
        },
        {
            "id": "sample_2",
            "question": "What did Marie Curie discover?",
            "context": "Marie Curie discovered the elements polonium and radium. She won Nobel Prizes in Physics and Chemistry.",
            "answer": "polonium and radium",
        },
        {
            "id": "sample_3",
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet was written by William Shakespeare. It is one of his most famous tragedies.",
            "answer": "William Shakespeare",
        },
    ]

    # Create data directory if it doesn't exist
    data_dir = ROOT_DIR / "data" / "json"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save sample data
    import json

    sample_file = data_dir / "sample_test.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Created sample data file: {sample_file}")
    return sample_file


def run_sample_pipeline():
    """Run the pipeline on sample data."""
    print("\n=== Running Sample Pipeline ===")

    # Create sample data
    sample_file = create_sample_data()

    # Run pipeline on sample data
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_schema_pipeline.py"),
        "--input",
        str(sample_file),
        "--dataset",
        "json",
        "--output",
        "output/sample_pipeline_test",
        "--compression-method",
        "faiss_max_size",
        "--max-size",
        "5",
        "--use-synonyms",
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)

        if result.returncode == 0:
            print("✓ Pipeline completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("✗ Pipeline failed!")
            print("Error:")
            print(result.stderr)

    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    run_pipeline_example()
    run_sample_pipeline()
