#!/usr/bin/env python3
"""
Example usage of FAISS-based schema compression with different options.
"""

from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.encoder import Encoder
from core.schema_definer import SchemaDefiner
from config import BASE_ENCODER_MODEL, SD_PROMPT_PATH, SD_FEW_SHOT_EXAMPLES_PATH


def example_faiss_compression():
    """Example showing FAISS-based schema compression options."""
    print("=== FAISS Schema Compression Example ===")
    
    # Initialize encoder and schema definer
    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL, framework="transformers")
    schema_definer = SchemaDefiner(
        model=encoder,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )
    
    # Example schema with many similar relations
    example_schema = {
        "born_in": "Indicates the place where someone was born",
        "birth_place": "The location where a person was born", 
        "place_of_birth": "Where someone came into the world",
        "birthplace": "The geographical location of birth",
        "discovered": "Found or identified something new",
        "found": "Located or came across something",
        "invented": "Created or developed something new",
        "created": "Brought something into existence",
        "developed": "Created or improved something over time",
        "won": "Achieved victory or received an award",
        "received": "Got or obtained something",
        "awarded": "Given recognition or prize",
        "obtained": "Acquired or gained possession of",
        "studied_at": "Attended an educational institution",
        "educated_at": "Received education from an institution",
        "graduated_from": "Completed studies at an institution",
        "attended": "Was present at or went to regularly",
        "works_at": "Is employed by or associated with",
        "employed_by": "Has a job or position with",
        "affiliated_with": "Connected or associated with",
    }
    
    print(f"Original schema: {len(example_schema)} relations")
    
    # Option 1: Compress to maximum schema size
    print(f"\n--- Option 1: Maximum Schema Size (10 relations) ---")
    compressed_max = schema_definer.compress_schema(
        example_schema,
        method="faiss_max_size",
        max_size=10
    )
    print(f"Compressed to {len(compressed_max)} relations:")
    for rel in compressed_max.keys():
        print(f"  - {rel}")
    
    # Option 2: Compress by ratio
    print(f"\n--- Option 2: Compression Ratio (40% of original) ---")
    compressed_ratio = schema_definer.compress_schema(
        example_schema,
        method="faiss_ratio", 
        compression_ratio=0.4
    )
    print(f"Compressed to {len(compressed_ratio)} relations:")
    for rel in compressed_ratio.keys():
        print(f"  - {rel}")
    
    # Option 3: Similarity-based grouping
    print(f"\n--- Option 3: Similarity Groups (threshold 0.6) ---")
    compressed_similarity = schema_definer.compress_schema(
        example_schema,
        method="faiss_similarity",
        threshold=0.6
    )
    print(f"Compressed to {len(compressed_similarity)} relations:")
    for rel in compressed_similarity.keys():
        print(f"  - {rel}")
    
    print(f"\n--- Summary ---")
    print(f"Original: {len(example_schema)} relations")
    print(f"Max size (10): {len(compressed_max)} relations")
    print(f"Ratio (40%): {len(compressed_ratio)} relations") 
    print(f"Similarity (0.6): {len(compressed_similarity)} relations")
    
    print("\n=== FAISS compression provides precise control over schema size! ===")


if __name__ == "__main__":
    example_faiss_compression()
