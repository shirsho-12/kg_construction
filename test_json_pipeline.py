#!/usr/bin/env python3
"""
Test script for JSON pipeline functionality.
"""
from pathlib import Path
import logging

from src.dataset import JSONDataset, GraphConstructionEvaluator
from src.qa_system import QASystem
from src.config import EXAMPLE_DATA_PATH_JSON

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_json_datasets():
    """Test JSON dataset loading for both tasks."""
    print("Testing JSON Dataset Loading")
    print("=" * 40)
    
    # Test graph construction dataset
    gc_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, task_type="graph_construction")
    print(f"Graph Construction Dataset: {len(gc_dataset)} samples")
    
    sample = gc_dataset[0]
    print(f"Sample type: {sample['type']}")
    print(f"Question: {sample['question']}")
    print(f"Context entities: {list(sample['context'].keys())}")
    print(f"Evidences: {sample['evidences']}")
    print()
    
    # Test QA dataset
    qa_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, task_type="qa")
    print(f"QA Dataset: {len(qa_dataset)} samples")
    
    sample = qa_dataset[0]
    print(f"Sample type: {sample['type']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Context preview: {sample['context'][:200]}...")
    print()


def test_graph_construction_evaluation():
    """Test graph construction evaluation."""
    print("Testing Graph Construction Evaluation")
    print("=" * 40)
    
    evaluator = GraphConstructionEvaluator()
    
    # Test cases
    test_cases = [
        {
            "predicted": [("Lothair II", "mother", "Ermengarde of Tours")],
            "ground_truth": [("Lothair II", "mother", "Ermengarde of Tours")],
            "description": "Perfect match"
        },
        {
            "predicted": [("Lothair II", "mother", "Ermengarde of Tours"), ("Lothair II", "father", "Unknown")],
            "ground_truth": [("Lothair II", "mother", "Ermengarde of Tours")],
            "description": "Extra prediction"
        },
        {
            "predicted": [("Lothair II", "mother", "Ermengarde of Tours")],
            "ground_truth": [("Lothair II", "mother", "Ermengarde of Tours"), ("Ermengarde of Tours", "date of death", "20 March 851")],
            "description": "Missing prediction"
        }
    ]
    
    for case in test_cases:
        scores = evaluator.evaluate_triplets(case["predicted"], case["ground_truth"])
        print(f"{case['description']}:")
        print(f"  Precision: {scores['precision']:.3f}")
        print(f"  Recall: {scores['recall']:.3f}")
        print(f"  F1: {scores['f1']:.3f}")
        print()


def test_qa_system():
    """Test QA system with example knowledge graph."""
    print("Testing QA System")
    print("=" * 40)
    
    qa_system = QASystem()
    
    # Load example knowledge graph
    example_triplets = [
        ("Lothair II", "mother", "Ermengarde of Tours"),
        ("Ermengarde of Tours", "date of death", "20 March 851"),
        ("Lothair II", "was married to", "Teutberga"),
        ("Teutberga", "date of death", "11 November 875"),
        ("Lothair II", "father", "Lothair I")
    ]
    
    qa_system.load_knowledge_graph(example_triplets)
    print(f"Loaded knowledge graph with {len(example_triplets)} triplets")
    
    # Test questions
    questions = [
        "When did Lothair II's mother die?",
        "Who was Lothair II's mother?",
        "When did Teutberga die?",
        "Who was Lothair II married to?",
        "Who was Lothair II's father?"
    ]
    
    for question in questions:
        result = qa_system.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['confidence']:.2f})")
        print()


def test_end_to_end():
    """Test end-to-end pipeline with real data."""
    print("Testing End-to-End Pipeline")
    print("=" * 40)
    
    # Load datasets
    gc_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, task_type="graph_construction")
    qa_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, task_type="qa")
    
    # Simulate extracted triplets (in real pipeline, this comes from OIE)
    simulated_triplets = [
        [("Lothair II", "mother", "Ermengarde of Tours"),
         ("Ermengarde of Tours", "date of death", "20 March 851")]
    ]
    
    # Evaluate graph construction
    evaluator = GraphConstructionEvaluator()
    graph_metrics = evaluator.evaluate_dataset(gc_dataset, simulated_triplets)
    
    print("Graph Construction Results:")
    for key, value in graph_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Test QA with simulated knowledge graph
    qa_system = QASystem()
    qa_system.load_knowledge_graph(simulated_triplets[0])
    
    sample = qa_dataset[0]
    question = sample["question"]
    ground_truth = sample["answer"]
    
    qa_result = qa_system.answer_question(question)
    
    print("QA Results:")
    print(f"  Question: {question}")
    print(f"  Predicted: {qa_result['answer']}")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Confidence: {qa_result['confidence']:.2f}")
    
    # Simple accuracy check
    is_correct = ground_truth.lower() in qa_result['answer'].lower()
    print(f"  Correct: {is_correct}")


if __name__ == "__main__":
    print("JSON Pipeline Test Suite")
    print("=" * 50)
    print()
    
    test_json_datasets()
    print()
    
    test_graph_construction_evaluation()
    print()
    
    test_qa_system()
    print()
    
    test_end_to_end()
    
    print("All tests completed!")
