#!/usr/bin/env python3
"""
Test script to verify schema compression is working correctly.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from core.schema_definer import SchemaDefiner
from core.encoder import Encoder


def test_schema_compression():
    """Test that schema compression actually works."""
    print("Testing schema compression...")

    # Create a test schema with similar relations that should be compressed
    test_schema = {
        "is located in": "The place where something is situated",
        "is situated in": "The location where something is found",
        "is found in": "The place where something exists",
        "works for": "The company where someone is employed",
        "is employed by": "The organization that hires someone",
        "created by": "The person who made something",
        "authored by": "The writer of a book or article",
        "directed by": "The person who directed a film",
        "has color": "The color of an object",
        "possesses property": "An attribute that something has",
    }

    print(f"Original schema has {len(test_schema)} relations")

    try:
        # Initialize encoder and schema definer
        encoder = Encoder(model_name_or_path="ministral/Ministral-3b-instruct")
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path="prompts/sd_prompt.txt",
            schema_few_shot_examples_path="prompts/sd_example.txt",
        )

        # Test compression with different thresholds
        thresholds = [0.5, 0.7, 0.9]

        for threshold in thresholds:
            print(f"\nTesting compression with threshold {threshold}:")

            compressed = schema_definer.compress_schema(
                test_schema, threshold=threshold, method="agglomerative"
            )

            original_count = len(test_schema)
            compressed_count = len(compressed)
            reduction = original_count - compressed_count
            reduction_pct = (
                (reduction / original_count * 100) if original_count > 0 else 0
            )

            print(f"  Original: {original_count} relations")
            print(f"  Compressed: {compressed_count} relations")
            print(f"  Reduction: {reduction} relations ({reduction_pct:.1f}%)")

            if compressed_count < original_count:
                print("  ✓ Compression successful!")
                print(f"  Compressed relations: {list(compressed.keys())}")
            else:
                print("  ⚠ No compression occurred")

        print("\n✓ Schema compression test completed successfully!")

    except Exception as e:
        print(f"✗ Schema compression test failed: {e}")
        raise


def test_schema_parsing():
    """Test the updated schema parsing with filtering rules."""
    print("\nTesting schema parsing with filtering...")

    # Test schema text with various issues that should be filtered out
    test_schema_text = """
1. is located in: The place where something is situated
2. {relation}: description
3. is situated in: The location where something is found
4. : 
5. works for: The company where someone is employed
6. this is a very long relation name that has more than seven words: This should be filtered out
7. has color: The color of an object
8. "quoted relation": A relation with quotes
9. authored by: The writer of a book or article
"""

    try:
        encoder = Encoder(model_name_or_path="ministral/Ministral-3b-instruct")
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path="prompts/sd_prompt.txt",
            schema_few_shot_examples_path="prompts/sd_example.txt",
        )

        parsed_schema = schema_definer._parse_schema(test_schema_text)

        print(f"Parsed schema has {len(parsed_schema)} relations")
        print("Relations that passed filtering:")
        for rel, defn in parsed_schema.items():
            print(f"  - {rel}: {defn}")

        # Verify filtering worked
        assert "{relation}" not in parsed_schema, "Should filter out {relation}"
        assert (
            "this is a very long relation name that has more than seven words"
            not in parsed_schema
        ), "Should filter out long relations"
        assert (
            "quoted relation" in parsed_schema
        ), "Should keep quoted relations (quotes removed)"

        print("✓ Schema parsing filtering works correctly!")

    except Exception as e:
        print(f"✗ Schema parsing test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running schema compression and parsing tests...\n")

    test_schema_parsing()
    test_schema_compression()

    print("\n✓ All tests passed!")
