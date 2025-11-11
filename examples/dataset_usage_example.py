#!/usr/bin/env python3
"""
Example script demonstrating the usage of the new JSON dataset classes.
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets import TwoWikiMultiHopQADataset, HotpotQADataset


def demo_two_wiki_multihopqa():
    """Demonstrate TwoWikiMultiHopQA dataset usage."""
    print("=== TwoWikiMultiHopQA Dataset Demo ===")
    
    # Example path - adjust to your actual data location
    data_path = Path("data/json/2wikimultihopqa.json")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please update the path to your actual 2wikimultihopqa.json file")
        return
    
    # Load dataset for graph construction
    dataset = TwoWikiMultiHopQADataset(data_path, task_type="graph_construction")
    
    print(f"Loaded {len(dataset)} samples")
    
    # Show dataset stats
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Show first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample (graph construction):")
        print(f"  ID: {sample['_id']}")
        print(f"  Type: {sample['type']}")
        print(f"  Question: {sample['question']}")
        print(f"  Context entities: {list(sample['context'].keys())}")
        print(f"  Evidences: {len(sample['evidences'])} triplets")
        print(f"  Answer: {sample['answer']}")
    
    # Load same dataset for QA
    qa_dataset = TwoWikiMultiHopQADataset(data_path, task_type="qa")
    
    if len(qa_dataset) > 0:
        qa_sample = qa_dataset[0]
        print(f"\nFirst sample (QA):")
        print(f"  ID: {qa_sample['_id']}")
        print(f"  Question: {qa_sample['question']}")
        print(f"  Context length: {len(qa_sample['context'])} characters")
        print(f"  Answer: {qa_sample['answer']}")


def demo_hotpotqa():
    """Demonstrate HotpotQA dataset usage."""
    print("\n=== HotpotQA Dataset Demo ===")
    
    # Example path - adjust to your actual data location
    data_path = Path("data/json/hotpotqa.json")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please update the path to your actual hotpotqa.json file")
        return
    
    # Load dataset for graph construction
    dataset = HotpotQADataset(data_path, task_type="graph_construction")
    
    print(f"Loaded {len(dataset)} samples")
    
    # Show dataset stats
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Show first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample (graph construction):")
        print(f"  ID: {sample['_id']}")
        print(f"  Type: {sample['type']}")
        print(f"  Level: {sample['level']}")
        print(f"  Question: {sample['question']}")
        print(f"  Context entities: {list(sample['context'].keys())}")
        print(f"  Answer: {sample['answer']}")
    
    # Show samples by difficulty level
    if len(dataset) > 0:
        levels = dataset.get_difficulty_levels()
        print(f"\nDifficulty levels: {levels}")
        
        for level in levels[:2]:  # Show first 2 levels
            samples = dataset.get_samples_by_level(level)
            print(f"  {level}: {len(samples)} samples")
    
    # Show samples by question type
    if len(dataset) > 0:
        types = dataset.get_question_types()
        print(f"\nQuestion types: {types}")
        
        for q_type in types[:2]:  # Show first 2 types
            samples = dataset.get_samples_by_type(q_type)
            print(f"  {q_type}: {len(samples)} samples")


if __name__ == "__main__":
    demo_two_wiki_multihopqa()
    demo_hotpotqa()
    
    print("\n=== Usage in Pipelines ===")
    print("To use these datasets in your pipelines:")
    print("1. Import: from datasets import TwoWikiMultiHopQADataset, HotpotQADataset")
    print("2. Initialize: dataset = TwoWikiMultiHopQADataset(path, task_type='graph_construction')")
    print("3. Use like any PyTorch dataset: for sample in dataset: ...")
    print("4. Access specific methods: stats = dataset.get_stats()")
