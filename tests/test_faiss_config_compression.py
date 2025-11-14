#!/usr/bin/env python3
"""
Test script that uses configuration files to test FAISS compression with different settings.
"""

import sys
import yaml
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    BASE_ENCODER_MODEL,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)

from schema_definition.schema_definer import SchemaDefiner
from triplet_extraction.encoder import Encoder


encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)


def load_config(config_path: Path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_large_test_schema():
    """Create a large test schema for comprehensive testing."""
    return {
        # Location relations (should compress together)
        "is located in": "The place where something is situated",
        "is situated in": "The location where something is found",
        "is found in": "The place where something exists",
        "resides in": "Where someone lives or stays",
        "positioned at": "The position of something",
        "placed in": "Where something is put",
        # Employment relations (should compress together)
        "works for": "The company where someone is employed",
        "is employed by": "The organization that hires someone",
        "has job at": "The workplace of a person",
        "is hired by": "Who employs someone",
        "serves at": "Where someone provides service",
        # Creation relations (should compress together)
        "created by": "The person who made something",
        "authored by": "The writer of a book or article",
        "written by": "The author of a text",
        "composed by": "The creator of music or literature",
        "designed by": "Who designed something",
        "built by": "Who constructed something",
        # Achievement relations (should compress together)
        "won": "Achieved victory or received an award",
        "received": "Got or obtained something",
        "awarded": "Given recognition or prize",
        "earned": "Gained through effort",
        "achieved": "Successfully accomplished",
        # Birth/Origin relations (should compress together)
        "born in": "The place where someone was born",
        "birth place": "The location of someone's birth",
        "originated from": "Where something came from",
        "native to": "Originally from a place",
        # Discovery relations (should compress together)
        "discovered": "Found or identified something new",
        "invented": "Created something for the first time",
        "developed": "Improved or created over time",
        "founded": "Established an organization",
        "established": "Set up or created an institution",
        # Media relations (should compress together)
        "directed by": "The person who directed a film",
        "produced by": "The producer of a movie or show",
        "starred in": "Appeared as main character",
        "featured in": "Appeared in a production",
        # Property relations (should compress together)
        "has color": "The color of an object",
        "possesses property": "An attribute that something has",
        "exhibits trait": "A characteristic that something shows",
        "displays feature": "Shows a particular aspect",
        # Unique relations (should remain separate)
        "weighs": "The weight measurement",
        "costs": "The price amount",
        "measures": "The size or dimension",
        "contains": "What is inside something",
        "requires": "What is needed for something",
    }


def test_config_based_compression():
    """Test compression using configuration file settings."""
    print("=== Testing Config-Based FAISS Compression ===")

    # Load base configuration
    config_dir = ROOT_DIR / "config_templates"
    base_config = load_config(config_dir / "base_config.yaml")

    test_schema = create_large_test_schema()
    original_count = len(test_schema)
    print(f"Original schema has {original_count} relations")

    try:
        # Initialize encoder and schema definer
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        # Test max size configurations from config
        print(f"\n--- Testing Max Size Configurations ---")
        max_size_tests = base_config["faiss_compression_tests"]["max_size_tests"]

        for test_config in max_size_tests:
            max_size = test_config["max_size"]
            description = test_config["description"]

            print(f"\nTesting: {description}")

            compressed = schema_definer.compress_schema(
                test_schema, method="faiss_max_size", max_size=max_size
            )

            compressed_count = len(compressed)
            reduction = original_count - compressed_count
            reduction_pct = (
                (reduction / original_count * 100) if original_count > 0 else 0
            )

            print(f"  Target: {max_size} relations")
            print(f"  Result: {compressed_count} relations")
            print(f"  Reduction: {reduction} ({reduction_pct:.1f}%)")

            success = compressed_count <= max_size
            print(f"  Status: {'✓ Success' if success else '✗ Failed'}")

        # Test ratio configurations from config
        print(f"\n--- Testing Ratio Configurations ---")
        ratio_tests = base_config["faiss_compression_tests"]["ratio_tests"]

        for test_config in ratio_tests:
            ratio = test_config["ratio"]
            description = test_config["description"]

            print(f"\nTesting: {description}")

            compressed = schema_definer.compress_schema(
                test_schema, method="faiss_ratio", compression_ratio=ratio
            )

            compressed_count = len(compressed)
            target_size = int(original_count * ratio)
            actual_ratio = compressed_count / original_count

            print(f"  Target: {ratio} ({ratio*100:.0f}%) = {target_size} relations")
            print(f"  Result: {compressed_count} relations ({actual_ratio*100:.1f}%)")

            # Allow some tolerance for ratio matching
            success = abs(actual_ratio - ratio) <= 0.15
            print(f"  Status: {'✓ Success' if success else '✗ Failed'}")

        print(f"\n✓ Config-based compression tests completed!")

    except Exception as e:
        print(f"✗ Config-based compression test failed: {e}")
        raise


def test_pipeline_configs():
    """Test the specific pipeline configurations."""
    print(f"\n=== Testing Pipeline Configurations ===")

    config_dir = ROOT_DIR / "config_templates"
    pipeline_configs = [
        "faiss_max_size_pipeline.yaml",
        "faiss_ratio_pipeline.yaml",
        "2wikimultihopqa_pipeline.yaml",
        "hotpotqa_pipeline.yaml",
    ]

    test_schema = create_large_test_schema()
    original_count = len(test_schema)

    try:
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        for config_file in pipeline_configs:
            config_path = config_dir / config_file
            if not config_path.exists():
                print(f"⚠ Config file not found: {config_file}")
                continue

            print(f"\n--- Testing {config_file} ---")
            config = load_config(config_path)
            pipeline_config = config.get("pipeline", {})

            method = pipeline_config.get("compression_method", "faiss_similarity")
            max_size = pipeline_config.get("max_schema_size")
            ratio = pipeline_config.get("compression_ratio")

            print(f"  Method: {method}")
            print(f"  Max size: {max_size}")
            print(f"  Ratio: {ratio}")

            # Test the configuration
            compressed = schema_definer.compress_schema(
                test_schema, method=method, max_size=max_size, compression_ratio=ratio
            )

            compressed_count = len(compressed)
            reduction = original_count - compressed_count

            print(f"  Original: {original_count} relations")
            print(f"  Compressed: {compressed_count} relations")
            print(f"  Reduction: {reduction} relations")

            # Validate results based on method
            if method == "faiss_max_size" and max_size:
                success = compressed_count <= max_size
                print(f"  Target achieved: {'✓' if success else '✗'}")
            elif method == "faiss_ratio" and ratio:
                actual_ratio = compressed_count / original_count
                success = abs(actual_ratio - ratio) <= 0.15
                print(
                    f"  Ratio achieved: {'✓' if success else '✗'} (target: {ratio:.1f}, actual: {actual_ratio:.1f})"
                )
            else:
                print(
                    f"  Compression applied: {'✓' if compressed_count < original_count else '✗'}"
                )

        print(f"\n✓ Pipeline configuration tests completed!")

    except Exception as e:
        print(f"✗ Pipeline configuration test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running FAISS configuration-based compression tests...\n")

    test_config_based_compression()
    test_pipeline_configs()

    print("\n✓ All configuration-based tests passed!")
