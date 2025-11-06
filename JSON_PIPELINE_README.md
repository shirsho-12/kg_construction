# JSON Pipeline Implementation

This implementation provides a complete pipeline for handling structured JSON datasets for both graph construction and question answering tasks.

## Overview

The JSON pipeline supports four question types:
- **compositional**: Multi-step reasoning questions
- **comparison**: Questions comparing entities
- **inference**: Questions requiring inference
- **bridge_comparison**: Complex comparison questions

## Components

### 1. JSONDataset (`src/dataset.py`)

**Purpose**: Load and format structured JSON data for different tasks.

**Features**:
- Supports both graph construction and QA tasks
- Converts nested context to `entity: sentences` format for graph construction
- Converts to plain text format for QA tasks
- Handles evidences for evaluation

**Usage**:
```python
from src.dataset import JSONDataset

# For graph construction
gc_dataset = JSONDataset(data_path, task_type="graph_construction")

# For QA
qa_dataset = JSONDataset(data_path, task_type="qa")
```

### 2. GraphConstructionEvaluator (`src/dataset.py`)

**Purpose**: Evaluate extracted triplets against ground truth evidences.

**Metrics**:
- Precision, Recall, F1 scores
- Per-type evaluation (compositional, comparison, etc.)
- Overall dataset metrics

**Usage**:
```python
from src.dataset import GraphConstructionEvaluator

evaluator = GraphConstructionEvaluator()
scores = evaluator.evaluate_triplets(predicted, ground_truth)
```

### 3. QASystem (`src/qa_system.py`)

**Purpose**: Answer questions using extracted knowledge graphs.

**Features**:
- Pattern-based question answering
- Confidence scoring
- Supporting triplet extraction
- Simple evaluation metrics

**Usage**:
```python
from src.qa_system import QASystem

qa_system = QASystem()
qa_system.load_knowledge_graph(triplets)
result = qa_system.answer_question(question)
```

### 4. JSON Pipeline (`src/json_pipeline.py`)

**Purpose**: End-to-end pipeline integrating all components.

**Workflow**:
1. Load JSON dataset
2. Extract triplets using OIE
3. Apply schema compression
4. Evaluate graph construction
5. Run QA evaluation
6. Generate comprehensive reports

## Data Format

### Input JSON Structure
```json
{
  "_id": "unique_id",
  "type": "compositional|comparison|inference|bridge_comparison",
  "question": "Question text",
  "context": [
    ["Entity1", ["Sentence1", "Sentence2"]],
    ["Entity2", ["Sentence3", "Sentence4"]]
  ],
  "evidences": [
    ["Entity1", "relation", "Entity2"],
    ["Entity2", "relation", "value"]
  ],
  "answer": "Ground truth answer"
}
```

### Graph Construction Output
- **Context**: `{entity: "combined sentences"}`
- **Evidences**: `[entity, relation, entity]` triplets for evaluation

### QA Output
- **Context**: Plain text with entity labels
- **Question-Answer pairs** for evaluation

## Evaluation

### Graph Construction Metrics
- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total ground truth
- **F1**: Harmonic mean of precision and recall
- **Per-type metrics**: Separate scores for each question type

### QA Metrics
- **Accuracy**: Correct answers / Total questions
- **High-confidence accuracy**: Performance on confident predictions
- **Supporting evidence**: Triplets used for answering

## Usage Examples

### Basic Dataset Loading
```python
from src.dataset import JSONDataset
from src.config import EXAMPLE_DATA_PATH_JSON

# Load for graph construction
gc_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, "graph_construction")
sample = gc_dataset[0]
print(f"Entities: {list(sample['context'].keys())}")
print(f"Evidences: {sample['evidences']}")

# Load for QA
qa_dataset = JSONDataset(EXAMPLE_DATA_PATH_JSON, "qa")
sample = qa_dataset[0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
```

### Graph Construction Evaluation
```python
from src.dataset import GraphConstructionEvaluator

evaluator = GraphConstructionEvaluator()
predicted = [("Lothair II", "mother", "Ermengarde of Tours")]
ground_truth = [("Lothair II", "mother", "Ermengarde of Tours")]

scores = evaluator.evaluate_triplets(predicted, ground_truth)
print(f"F1 Score: {scores['f1']:.3f}")
```

### Question Answering
```python
from src.qa_system import QASystem

qa_system = QASystem()
triplets = [
    ("Lothair II", "mother", "Ermengarde of Tours"),
    ("Ermengarde of Tours", "date of death", "20 March 851")
]
qa_system.load_knowledge_graph(triplets)

result = qa_system.answer_question("When did Lothair II's mother die?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Complete Pipeline
```python
from src.json_pipeline import run_json_pipeline
from pathlib import Path

metrics = run_json_pipeline(
    data_path=Path("data/json/test.json"),
    output_dir=Path("output/json_pipeline"),
    use_synonyms=True,
    compression_method="agglomerative",
    compression_threshold=0.8
)

graph_metrics, qa_metrics = metrics
print(f"Graph F1: {graph_metrics['overall_f1']:.3f}")
print(f"QA Accuracy: {qa_metrics['accuracy']:.3f}")
```

## Testing

Run the comprehensive test suite:
```bash
python test_json_pipeline.py
```

This tests:
- Dataset loading for both tasks
- Graph construction evaluation
- QA system functionality
- End-to-end pipeline integration

## File Structure

```
src/
├── dataset.py          # JSONDataset and GraphConstructionEvaluator
├── qa_system.py        # QASystem implementation
├── json_pipeline.py    # End-to-end pipeline
└── config.py          # Configuration with JSON paths

test_json_pipeline.py  # Comprehensive test suite
JSON_PIPELINE_README.md # This documentation
```

## Key Features

✅ **Proper JSON loading** with error handling
✅ **Entity formatting** for graph construction (`entity: sentences`)
✅ **Context formatting** for QA (plain text)
✅ **Evidence-based evaluation** using semantic matching
✅ **Question answering** using knowledge graphs
✅ **Comprehensive metrics** for both tasks
✅ **Support for all 4 question types**
✅ **Extensible design** for enhancements

## Future Enhancements

- Semantic relation matching using embeddings
- Advanced QA with reasoning chains
- Batch processing for large datasets
- Integration with external knowledge bases
- Interactive visualization of knowledge graphs
