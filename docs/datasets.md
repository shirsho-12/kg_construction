# JSON Dataset Classes

This document describes the JSON dataset classes for handling different dataset formats in the KG construction pipeline.

## Overview

The dataset system is organized with a base class and specific implementations for each dataset type:

- `BaseJSONDataset`: Abstract base class with common functionality
- `TwoWikiMultiHopQADataset`: Implementation for 2WikiMultiHopQA data format
- `HotpotQADataset`: Implementation for HotpotQA data format

## BaseJSONDataset

The base class provides common functionality for all JSON datasets:

### Features
- Loading JSON data from files
- Handling both single objects and arrays
- Support for graph construction and QA tasks
- Common methods for accessing samples by ID
- Context processing utilities

### Methods
- `__init__(data_path, task_type)`: Initialize dataset
- `get_sample_by_id(sample_id)`: Get sample by its _id field
- `get_all_ids()`: Get all _id values in dataset
- `_process_context(context)`: Process context from [entity, [sentences]] format

## TwoWikiMultiHopQADataset

Handles the 2WikiMultiHopQA dataset format.

### Data Format
```json
{
  "_id": "primary_key",
  "type": "compositional|argumentative|...",
  "question": "Question text",
  "context": [
    ["Entity A", ["Sentence 1", "Sentence 2"]],
    ["Entity B", ["Sentence 1", "Sentence 2"]]
  ],
  "evidences": [
    ["Entity A", "relation", "Entity B"],
    ["Entity B", "relation", "Entity C"]
  ],
  "answer": "Answer text"
}
```

### Specific Methods
- `get_evidence_triplets(sample_idx)`: Extract evidence triplets for evaluation
- `get_question_types()`: Get all unique question types
- `get_stats()`: Get dataset statistics

### Usage Example
```python
from datasets import TwoWikiMultiHopQADataset

# Load for graph construction
dataset = TwoWikiMultiHopQADataset("data/2wikimultihopqa.json", "graph_construction")
sample = dataset[0]
print(f"Context entities: {list(sample['context'].keys())}")
print(f"Evidences: {sample['evidences']}")

# Load for QA
qa_dataset = TwoWikiMultiHopQADataset("data/2wikimultihopqa.json", "qa")
qa_sample = qa_dataset[0]
print(f"Full context: {qa_sample['context']}")

# Get statistics
stats = dataset.get_stats()
print(f"Question types: {stats['question_types']}")
```

## HotpotQADataset

Handles the HotpotQA dataset format.

### Data Format
```json
{
  "_id": "primary_key",
  "question": "Question text",
  "answer": "Answer text",
  "context": [
    ["Entity A", ["Sentence 1", "Sentence 2"]],
    ["Entity B", ["Sentence 1", "Sentence 2"]]
  ],
  "type": "factual|comparative|...",
  "level": "easy|medium|hard|..."
}
```

### Specific Methods
- `get_question_types()`: Get all unique question types
- `get_difficulty_levels()`: Get all unique difficulty levels
- `get_samples_by_level(level)`: Get samples by difficulty level
- `get_samples_by_type(question_type)`: Get samples by question type
- `get_stats()`: Get dataset statistics

### Usage Example
```python
from datasets import HotpotQADataset

# Load dataset
dataset = HotpotQADataset("data/hotpotqa.json", "graph_construction")

# Get samples by difficulty
easy_samples = dataset.get_samples_by_level("easy")
print(f"Easy samples: {len(easy_samples)}")

# Get samples by type
factual_samples = dataset.get_samples_by_type("factual")
print(f"Factual questions: {len(factual_samples)}")

# Get statistics
stats = dataset.get_stats()
print(f"Difficulty levels: {stats['difficulty_levels']}")
```

## Task Types

### Graph Construction
- Returns context as entity: sentences dictionary
- Includes evidences for evaluation (2WikiMultiHopQA)
- Optimized for triplet extraction and schema definition

### QA
- Returns context as single combined string
- Includes entity labels in context
- Optimized for question answering

## Integration with Pipelines

These datasets can be used directly in the existing pipelines:

```python
# In json_pipeline.py
from datasets import TwoWikiMultiHopQADataset, HotpotQADataset

# Load appropriate dataset based on data format
if data_format == "2wikimultihopqa":
    graph_dataset = TwoWikiMultiHopQADataset(data_path, "graph_construction")
    qa_dataset = TwoWikiMultiHopQADataset(data_path, "qa")
elif data_format == "hotpotqa":
    graph_dataset = HotpotQADataset(data_path, "graph_construction")
    qa_dataset = HotpotQADataset(data_path, "qa")
```

## Testing

Run the test suite to verify functionality:

```bash
python tests/test_datasets.py
```

## File Structure

```
src/datasets/
├── __init__.py
├── base_json_dataset.py
├── two_wiki_multihopqa_dataset.py
├── hotpotqa_dataset.py
├── json_dataset.py (existing)
└── text_dataset.py (existing)

examples/
└── dataset_usage_example.py

tests/
└── test_datasets.py
```
