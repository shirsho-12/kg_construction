# KG Construction

A comprehensive knowledge graph construction pipeline with support for both text and JSON datasets, featuring schema definition, compression, and QA evaluation.

## Features

- **Dual Pipeline Support**: Process both raw text and structured JSON datasets
- **Schema Definition & Compression**: Automatic relation extraction with clustering-based compression
- **Synonym Generation**: Enhanced entity matching through synonym extraction
- **QA Evaluation**: Two evaluation methods - word matching and Graph RAG
- **Schema Refinement**: Iterative schema improvement without re-running extraction
- **YAML Configuration**: Flexible configuration system for all pipeline modes

## Project Structure

```
kg_construction/
├── src/
│   ├── config.py                 # Configuration constants
│   ├── main.py                   # Main entry point with YAML config support
│   ├── core/                     # Core processing components
│   │   ├── encoder.py           # Language model encoder
│   │   ├── extractor.py         # Text extraction utilities
│   │   ├── oie.py               # Open Information Extraction
│   │   └── schema_definer.py    # Schema definition and compression
│   ├── pipelines/                # Pipeline implementations
│   │   ├── pipeline.py          # Text dataset pipeline
│   │   ├── json_pipeline.py     # JSON dataset pipeline
│   │   ├── pipeline_utils.py    # Shared utilities
│   │   └── schema_refiner.py    # Schema refinement tool
│   ├── evaluation/               # Evaluation components
│   │   └── qa_system.py         # QA system with dual evaluation methods
│   └── datasets/                 # Dataset handling
│       ├── text_dataset.py      # Text dataset class
│       ├── json_dataset.py      # JSON dataset classes
│       └── evaluator.py         # Graph construction evaluator
├── configs/                      # YAML configuration files
│   ├── text_pipeline.yaml
│   ├── json_pipeline.yaml
│   ├── schema_refinement.yaml
│   └── qa_evaluation.yaml
├── data/                         # Input data
│   ├── text/
│   └── json/
├── output/                       # Pipeline outputs
├── prompts/                      # LLM prompts
└── model_cache/                  # Cached models
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd kg_construction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have the required prompts in the `prompts/` directory:
   - `oie_prompt.txt`
   - `oie_synonyms_prompt.txt`
   - `oie_example.txt`
   - `oie_example_synonyms.txt`
   - `sd_prompt.txt`
   - `sd_example.txt`

## Quick Start

### 1. Create Configuration Files

Generate example configuration files:

```bash
python src/main.py --create-configs
```

This creates four YAML files in the `configs/` directory that you can customize.

### 2. Run Pipelines

#### Text Dataset Pipeline

```bash
python src/main.py --config configs/text_pipeline.yaml --mode text
```

#### JSON Dataset Pipeline

```bash
python src/main.py --config configs/json_pipeline.yaml --mode json
```

#### Schema Refinement

```bash
python src/main.py --config configs/schema_refinement.yaml --mode schema_refiner
```

#### QA Evaluation

```bash
python src/main.py --config configs/qa_evaluation.yaml --mode qa_evaluation
```

## Configuration

### Text Pipeline Configuration

```yaml
pipeline:
  data_path: "data/text/example.txt" # Input text file
  output_dir: "output/text/example" # Output directory
  use_synonyms: true # Enable synonym generation
  compression_method: "faiss_similarity" # Compression method
  compression_threshold: 0.8 # Compression threshold
  compress_if_more_than: 30 # Min relations for compression

logging:
  level: "INFO" # Logging level
```

### JSON Pipeline Configuration

```yaml
pipeline:
  data_path: "data/json/test.json" # Input JSON file
  output_dir: "output/json/example" # Output directory
  use_synonyms: true # Enable synonym generation
  compression_method: "faiss_similarity" # Compression method
  compression_threshold: 0.8 # Compression threshold
  compress_if_more_than: 30 # Min relations for compression
  extraction_mode: "base" # Extraction mode: base/chunking/sentence
  chunk_size: 100 # Chunk size for chunking mode

logging:
  level: "INFO" # Logging level
```

### Schema Refinement Configuration

```yaml
schema_refiner:
  output_dir: "output/json/example" # Directory with previous results
  variant: "triplets_synonyms_text" # Refinement variant
  # Variants:
  # - "triplets_only": Use only extracted triplets
  # - "triplets_text": Use triplets + input text
  # - "triplets_synonyms_text": Use triplets + synonyms + input text

  triplets_file: "triplets.json" # Triplets file name
  synonyms_file: "synonyms.json" # Synonyms file name
  input_file: "results.json" # Input file with texts
  compression_method: "faiss_similarity" # Compression method
  compression_threshold: 0.8 # Compression threshold
  compress_if_more_than: 30 # Min relations for compression

logging:
  level: "INFO" # Logging level
```

### QA Evaluation Configuration

```yaml
qa_evaluation:
  data_path: "data/json/test.json" # Test dataset
  triplets_file: "output/json/example/triplets.json" # Knowledge graph file
  output_dir: "output/json/example" # Results output directory

logging:
  level: "INFO" # Logging level
```

## Pipeline Modes

### 1. Text Pipeline

Processes raw text files to extract entities, relations, and build knowledge graphs.

**Key Features:**

- Open Information Extraction (OIE)
- Synonym generation for entity matching
- Schema definition and compression
- Knowledge graph construction

### 2. JSON Pipeline

Processes structured JSON datasets with entity contexts and questions.

**Key Features:**

- Entity-aware extraction (preserves entity boundaries)
- Multiple extraction modes:
  - `base`: Process each sample as a whole
  - `chunking`: Split samples into word chunks
  - `sentence`: Process sentence by sentence
- Complete isolation between different `_id` samples
- Graph construction evaluation using ground truth evidences
- QA system integration

### 3. Schema Refinement

Refines schema definitions without re-running entity extraction.

**Variants:**

- **triplets_only**: Uses only extracted triplets
- **triplets_text**: Uses triplets + original input text
- **triplets_synonyms_text**: Uses triplets + synonyms + input text

**Benefits:**

- Faster iteration on schema definition
- No need to re-run expensive extraction
- Flexible input combinations

### 4. QA Evaluation

Evaluates question answering performance using two methods.

**Methods:**

- **word_match**: Pattern-based answer extraction
- **graph_rag**: Language model queries the knowledge graph

**Output:**

- Accuracy metrics for both methods
- Detailed results with questions and answers
- Comparison between evaluation approaches

## Data Formats

### Text Input

Plain text files with one document per line or standard paragraph format.

### JSON Input Format

```json
[
  {
    "_id": "sample_1",
    "type": "graph_construction",
    "context": [
      ["Entity1", ["Sentence1 about Entity1.", "Sentence2 about Entity1."]],
      ["Entity2", ["Sentence1 about Entity2.", "Sentence2 about Entity2."]]
    ],
    "question": "What is the relationship between Entity1 and Entity2?",
    "answer": "Entity1 is related to Entity2 through...",
    "evidences": [["Entity1", "relation", "Entity2"]]
  }
]
```

## Output Files

### Common Outputs

- `triplets.json`: Extracted entity-relation-entity triplets
- `synonyms.json`: Generated synonyms for entities
- `results.json`: Complete processing results
- `schema_definitions.json`: Relation definitions
- `compressed_schema.json`: Compressed relations
- `compression_outcomes.json`: Before/after compression comparison

### JSON Pipeline Specific

- `triplets_by_id.json`: Triplets organized by sample `_id`
- `graph_construction_results.json`: Complete results by sample
- `problematic_cases.json`: Error tracking and debugging

### QA Evaluation

- `qa_results_word_match.json`: Word match evaluation results
- `qa_results_graph_rag.json`: Graph RAG evaluation results

## Advanced Usage

### Custom Extraction Modes

For JSON datasets, choose the extraction mode based on your data characteristics:

- **`base`**: Best for concise, focused contexts
- **`chunking`**: Good for long contexts, preserves entity relationships
- **`sentence`**: Ideal for sentence-level precision

### Compression Methods

- **faiss_similarity**: Uses FAISS for vector similarity clustering

### Schema Refinement Workflow

1. Run initial pipeline with basic settings
2. Review schema definitions and compression results
3. Run schema refinement with different variants
4. Compare outcomes and select best configuration
5. Optionally re-run full pipeline with refined settings

## Error Handling

The system includes comprehensive error handling:

- **Problematic case tracking**: All extraction errors are logged
- **Fallback file detection**: Automatically finds alternative file names
- **Graceful degradation**: Continues processing when individual samples fail
- **Detailed logging**: Configurable logging levels for debugging

## Performance Considerations

- **Model caching**: Models are cached in `model_cache/` to avoid re-downloading
- **Batch processing**: OIE extraction uses batching for efficiency
- **Chunking strategies**: Large texts are automatically chunked for processing
- **Memory management**: Progressive processing to handle large datasets

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the project root
2. **Missing prompts**: Check that all prompt files exist in `prompts/`
3. **Memory issues**: Reduce batch size or use chunking for large texts
4. **Poor extraction quality**: Adjust compression thresholds and review prompts

### Debug Mode

Set logging level to `DEBUG` in your config for detailed execution information:

```yaml
logging:
  level: "DEBUG"
```

## Datasets

- [WebNLG](https://synalp.gitlabpages.inria.fr/webnlg-challenge/), [HF Link](https://huggingface.co/datasets/GEM/web_nlg)
- [WikiNRE](https://huggingface.co/datasets/Saibo-creator/wiki-nre)

## Model Notes

Note: "ministral/Ministral-3b-instruct" may not work well for schema definition. Use "mistralai/Mistral-7B-Instruct-v0.2" for better results.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
