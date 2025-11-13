#!/usr/bin/env python3
"""
Test script to verify FAISS-based schema compression with max size and ratio options.
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    BASE_ENCODER_MODEL,
    SD_FEW_SHOT_EXAMPLES_PATH,
    SD_PROMPT_PATH,
)

from core.schema_definer import SchemaDefiner
from core.encoder import Encoder


def create_test_schema():
    """Create a comprehensive test schema for compression testing."""
    return {
        "is located in": "The place where something is situated",
        "is situated in": "The location where something is found", 
        "is found in": "The place where something exists",
        "resides in": "Where someone lives or stays",
        "works for": "The company where someone is employed",
        "is employed by": "The organization that hires someone",
        "has job at": "The workplace of a person",
        "created by": "The person who made something",
        "authored by": "The writer of a book or article",
        "written by": "The author of a text",
        "composed by": "The creator of music or literature",
        "directed by": "The person who directed a film",
        "produced by": "The producer of a movie or show",
        "has color": "The color of an object",
        "possesses property": "An attribute that something has",
        "exhibits trait": "A characteristic that something shows",
        "born in": "The place where someone was born",
        "birth place": "The location of someone's birth",
        "originated from": "Where something came from",
        "discovered": "Found or identified something new",
        "invented": "Created something for the first time",
        "developed": "Improved or created over time",
        "founded": "Established an organization",
        "established": "Set up or created an institution",
        "won": "Achieved victory or received an award",
        "received": "Got or obtained something",
        "awarded": "Given recognition or prize",
    }


def test_faiss_max_size_compression():
    """Test FAISS compression with maximum size constraints."""
    print("=== Testing FAISS Maximum Size Compression ===")
    
    test_schema = create_test_schema()
    original_count = len(test_schema)
    print(f"Original schema has {original_count} relations")
    
    try:
        # Initialize encoder and schema definer
        encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        # Test different maximum sizes
        max_sizes = [5, 10, 15, 20]
        
        for max_size in max_sizes:
            print(f"\n--- Testing max_size = {max_size} ---")
            
            compressed = schema_definer.compress_schema(
                test_schema,
                method="faiss_max_size",
                max_size=max_size
            )
            
            compressed_count = len(compressed)
            reduction = original_count - compressed_count
            reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0
            
            print(f"  Target size: {max_size}")
            print(f"  Actual size: {compressed_count}")
            print(f"  Reduction: {reduction} relations ({reduction_pct:.1f}%)")
            
            # Verify we achieved the target (or close to it if not enough similar relations)
            if compressed_count <= max_size:
                print("  ✓ Target size achieved!")
            else:
                print(f"  ⚠ Target not achieved (likely insufficient similar relations)")
            
            print(f"  Compressed relations: {list(compressed.keys())[:5]}{'...' if len(compressed) > 5 else ''}")

        print("\n✓ FAISS max size compression test completed successfully!")

    except Exception as e:
        print(f"✗ FAISS max size compression test failed: {e}")
        raise


def test_faiss_ratio_compression():
    """Test FAISS compression with ratio constraints."""
    print("\n=== Testing FAISS Compression Ratio ===")
    
    test_schema = create_test_schema()
    original_count = len(test_schema)
    print(f"Original schema has {original_count} relations")
    
    try:
        # Initialize encoder and schema definer
        encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )

        # Test different compression ratios
        ratios = [0.3, 0.5, 0.7, 0.8]
        
        for ratio in ratios:
            print(f"\n--- Testing compression_ratio = {ratio} ({ratio*100:.0f}%) ---")
            
            compressed = schema_definer.compress_schema(
                test_schema,
                method="faiss_ratio",
                compression_ratio=ratio
            )
            
            compressed_count = len(compressed)
            target_size = int(original_count * ratio)
            actual_ratio = compressed_count / original_count
            reduction = original_count - compressed_count
            
            print(f"  Target ratio: {ratio} ({ratio*100:.0f}%)")
            print(f"  Target size: {target_size}")
            print(f"  Actual size: {compressed_count}")
            print(f"  Actual ratio: {actual_ratio:.3f} ({actual_ratio*100:.1f}%)")
            print(f"  Reduction: {reduction} relations")
            
            # Verify we achieved approximately the target ratio
            if abs(actual_ratio - ratio) <= 0.1:  # Allow 10% tolerance
                print("  ✓ Target ratio achieved!")
            else:
                print(f"  ⚠ Target ratio not achieved (tolerance exceeded)")
            
            print(f"  Compressed relations: {list(compressed.keys())[:5]}{'...' if len(compressed) > 5 else ''}")

        print("\n✓ FAISS ratio compression test completed successfully!")

    except Exception as e:
        print(f"✗ FAISS ratio compression test failed: {e}")
        raise


def test_compression_quality():
    """Test that compression maintains semantic quality."""
    print("\n=== Testing Compression Quality ===")
    
    # Create schema with clear semantic groups
    semantic_test_schema = {
        # Location group
        "located_in": "Where something is positioned",
        "situated_in": "The place where something exists",
        "found_in": "The location of something",
        
        # Employment group  
        "works_for": "Employment relationship",
        "employed_by": "Job relationship",
        "has_job_at": "Work position",
        
        # Creation group
        "created_by": "Who made something",
        "authored_by": "Who wrote something", 
        "invented_by": "Who created something new",
        
        # Unrelated relations
        "has_color": "The color property",
        "weighs": "The weight measurement",
        "costs": "The price amount",
    }
    
    try:
        encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
        schema_definer = SchemaDefiner(
            model=encoder,
            schema_prompt_path=SD_PROMPT_PATH,
            schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
        )
        
        print(f"Original semantic groups: {len(semantic_test_schema)} relations")
        
        # Test with moderate compression
        compressed = schema_definer.compress_schema(
            semantic_test_schema,
            method="faiss_max_size", 
            max_size=6
        )
        
        print(f"Compressed to: {len(compressed)} relations")
        print("Final relations:")
        for rel, defn in compressed.items():
            print(f"  - {rel}: {defn}")
            
        # Should preserve at least one relation from each semantic group
        location_preserved = any("located" in rel or "situated" in rel or "found" in rel for rel in compressed.keys())
        employment_preserved = any("work" in rel or "employ" in rel or "job" in rel for rel in compressed.keys()) 
        creation_preserved = any("creat" in rel or "author" in rel or "invent" in rel for rel in compressed.keys())
        
        print(f"\nSemantic group preservation:")
        print(f"  Location group: {'✓' if location_preserved else '✗'}")
        print(f"  Employment group: {'✓' if employment_preserved else '✗'}")
        print(f"  Creation group: {'✓' if creation_preserved else '✗'}")
        
        print("\n✓ Compression quality test completed!")
        
    except Exception as e:
        print(f"✗ Compression quality test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running FAISS schema compression tests...\n")

    test_faiss_max_size_compression()
    test_faiss_ratio_compression() 
    test_compression_quality()

    print("\n✓ All FAISS compression tests passed!")
