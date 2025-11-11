#!/usr/bin/env python3
"""
Test script for the new JSON dataset classes.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets import TwoWikiMultiHopQADataset, HotpotQADataset


def create_test_data():
    """Create test data files for testing."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Test 2WikiMultiHopQA data
    two_wiki_data = [
        {
            "_id": "test_001",
            "type": "compositional",
            "question": "What is the relationship between Entity A and Entity B?",
            "context": [
                ["Entity A", ["Sentence 1 about Entity A.", "Sentence 2 about Entity A."]],
                ["Entity B", ["Sentence 1 about Entity B.", "Sentence 2 about Entity B."]]
            ],
            "evidences": [
                ["Entity A", "related_to", "Entity B"],
                ["Entity B", "has_property", "Property X"]
            ],
            "answer": "Entity A is related to Entity B"
        },
        {
            "_id": "test_002", 
            "type": "argumentative",
            "question": "Which statement is correct about Entity C?",
            "context": [
                ["Entity C", ["Sentence 1 about Entity C."]]
            ],
            "evidences": [
                ["Entity C", "is_a", "Type Y"]
            ],
            "answer": "Entity C is Type Y"
        }
    ]
    
    # Test HotpotQA data
    hotpot_data = [
        {
            "_id": "hotpot_001",
            "question": "What is the capital of Country X?",
            "answer": "City Y",
            "context": [
                ["Country X", ["Country X is a nation.", "The capital of Country X is City Y."]],
                ["City Y", ["City Y is a large city.", "City Y has many landmarks."]]
            ],
            "type": "factual",
            "level": "easy"
        },
        {
            "_id": "hotpot_002",
            "question": "What connects Person A and Person B?",
            "answer": "Organization C",
            "context": [
                ["Person A", ["Person A works at Organization C."]],
                ["Person B", ["Person B also works at Organization C."]],
                ["Organization C", ["Organization C is a company."]]
            ],
            "type": "comparative", 
            "level": "medium"
        }
    ]
    
    # Write test files
    two_wiki_file = temp_dir / "2wikimultihopqa.json"
    hotpot_file = temp_dir / "hotpotqa.json"
    
    with open(two_wiki_file, 'w') as f:
        json.dump(two_wiki_data, f, indent=2)
    
    with open(hotpot_file, 'w') as f:
        json.dump(hotpot_data, f, indent=2)
    
    return temp_dir, two_wiki_file, hotpot_file


def test_two_wiki_multihopqa():
    """Test TwoWikiMultiHopQADataset."""
    print("Testing TwoWikiMultiHopQADataset...")
    
    temp_dir, two_wiki_file, _ = create_test_data()
    
    try:
        # Test graph construction mode
        dataset = TwoWikiMultiHopQADataset(two_wiki_file, task_type="graph_construction")
        
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        sample = dataset[0]
        assert sample["_id"] == "test_001"
        assert sample["type"] == "compositional"
        assert "Entity A" in sample["context"]
        assert "Entity B" in sample["context"]
        assert len(sample["evidences"]) == 2
        assert sample["answer"] == "Entity A is related to Entity B"
        
        # Test QA mode
        qa_dataset = TwoWikiMultiHopQADataset(two_wiki_file, task_type="qa")
        qa_sample = qa_dataset[0]
        assert qa_sample["_id"] == "test_001"
        assert "Entity A:" in qa_sample["context"]
        assert "Entity B:" in qa_sample["context"]
        
        # Test methods
        question_types = dataset.get_question_types()
        assert "compositional" in question_types
        assert "argumentative" in question_types
        
        evidences = dataset.get_evidence_triplets(0)
        assert len(evidences) == 2
        assert evidences[0] == ["Entity A", "related_to", "Entity B"]
        
        stats = dataset.get_stats()
        assert stats["total_samples"] == 2
        assert stats["num_question_types"] == 2
        
        print("✓ TwoWikiMultiHopQADataset tests passed")
        
    except Exception as e:
        print(f"✗ TwoWikiMultiHopQADataset test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_hotpotqa():
    """Test HotpotQADataset."""
    print("Testing HotpotQADataset...")
    
    temp_dir, _, hotpot_file = create_test_data()
    
    try:
        # Test graph construction mode
        dataset = HotpotQADataset(hotpot_file, task_type="graph_construction")
        
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        sample = dataset[0]
        assert sample["_id"] == "hotpot_001"
        assert sample["type"] == "factual"
        assert sample["level"] == "easy"
        assert "Country X" in sample["context"]
        assert "City Y" in sample["context"]
        assert sample["answer"] == "City Y"
        
        # Test QA mode
        qa_dataset = HotpotQADataset(hotpot_file, task_type="qa")
        qa_sample = qa_dataset[0]
        assert qa_sample["_id"] == "hotpot_001"
        assert "Country X:" in qa_sample["context"]
        assert "City Y:" in qa_sample["context"]
        
        # Test methods
        question_types = dataset.get_question_types()
        assert "factual" in question_types
        assert "comparative" in question_types
        
        difficulty_levels = dataset.get_difficulty_levels()
        assert "easy" in difficulty_levels
        assert "medium" in difficulty_levels
        
        easy_samples = dataset.get_samples_by_level("easy")
        assert len(easy_samples) == 1
        assert easy_samples[0]["_id"] == "hotpot_001"
        
        factual_samples = dataset.get_samples_by_type("factual")
        assert len(factual_samples) == 1
        assert factual_samples[0]["_id"] == "hotpot_001"
        
        stats = dataset.get_stats()
        assert stats["total_samples"] == 2
        assert stats["num_question_types"] == 2
        assert stats["num_difficulty_levels"] == 2
        
        print("✓ HotpotQADataset tests passed")
        
    except Exception as e:
        print(f"✗ HotpotQADataset test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_base_functionality():
    """Test base functionality common to both datasets."""
    print("Testing base functionality...")
    
    temp_dir, two_wiki_file, _ = create_test_data()
    
    try:
        dataset = TwoWikiMultiHopQADataset(two_wiki_file, task_type="graph_construction")
        
        # Test get_sample_by_id
        sample = dataset.get_sample_by_id("test_001")
        assert sample is not None
        assert sample["_id"] == "test_001"
        
        sample = dataset.get_sample_by_id("nonexistent")
        assert sample is None
        
        # Test get_all_ids
        ids = dataset.get_all_ids()
        assert len(ids) == 2
        assert "test_001" in ids
        assert "test_002" in ids
        
        print("✓ Base functionality tests passed")
        
    except Exception as e:
        print(f"✗ Base functionality test failed: {e}")
        raise
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running dataset tests...\n")
    
    test_base_functionality()
    test_two_wiki_multihopqa()
    test_hotpotqa()
    
    print("\n✓ All tests passed!")
