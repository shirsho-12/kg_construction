# Schema Pipeline Script Usage Guide

This document explains how to use the `run_schema_pipeline.py` script to extract triplets, define schemas, and compress them using FAISS methods.

## Overview

The script `scripts/run_schema_pipeline.py` provides a complete pipeline that:

1. **Loads a JSON dataset** using the appropriate Dataset class
2. **Extracts triplets** from text contexts using OIE (Open Information Extraction)
3. **Defines a schema** from the extracted triplets
4. **Compresses the schema** using FAISS-based methods
5. **Saves all outputs** including original schema, compressed schema, triplets, and metadata

## Usage

### Basic Command Structure

```bash
python scripts/run_schema_pipeline.py \
    --input <input_file.json> \
    --dataset <dataset_type> \
    --output <output_directory> \
    [additional options]
```

### Required Arguments

- `--input` / `-i`: Path to input JSON file
- `--dataset` / `-d`: Dataset type (`json`, `hotpotqa`, `2wikimultihopqa`)  
- `--output` / `-o`: Output directory for results

### Optional Arguments

- `--compression-method`: Compression method (`faiss_max_size`, `faiss_ratio`, `faiss_similarity`)
- `--max-size`: Maximum schema size for `faiss_max_size` method (default: 20)
- `--compression-ratio`: Compression ratio for `faiss_ratio` method (e.g., 0.5 for 50%)
- `--similarity-threshold`: Similarity threshold for `faiss_similarity` method (default: 0.8)
- `--max-samples`: Maximum number of samples to process (for testing)
- `--use-synonyms`: Enable synonym extraction (default: True)
- `--no-synonyms`: Disable synonym extraction

## Examples

### Example 1: HotpotQA with Max Size Compression

```bash
python scripts/run_schema_pipeline.py \
    --input data/json/hotpotqa.json \
    --dataset hotpotqa \
    --output output/hotpotqa_schema_pipeline \
    --compression-method faiss_max_size \
    --max-size 20 \
    --max-samples 50
```

### Example 2: Generic JSON with Ratio Compression

```bash
python scripts/run_schema_pipeline.py \
    --input data/json/sample_test.json \
    --dataset json \
    --output output/json_schema_pipeline \
    --compression-method faiss_ratio \
    --compression-ratio 0.6 \
    --no-synonyms
```

### Example 3: 2WikiMultiHopQA with Similarity Compression

```bash
python scripts/run_schema_pipeline.py \
    --input data/json/2wikimultihopqa.json \
    --dataset 2wikimultihopqa \
    --output output/2wiki_schema_pipeline \
    --compression-method faiss_similarity \
    --similarity-threshold 0.7
```

## Input Data Formats

### Generic JSON Format

For `--dataset json`, the input should be a JSON array of objects:

```json
[
  {
    "id": "sample_1",
    "question": "Where was Albert Einstein born?",
    "context": "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
    "answer": "Ulm, Germany"
  }
]
```

### HotpotQA Format

For `--dataset hotpotqa`, the context should be nested:

```json
[
  {
    "_id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "context": [
      ["Scott Derrickson", ["Scott Derrickson is an American director..."]],
      ["Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker..."]]
    ],
    "answer": "yes"
  }
]
```

## Output Structure

The script creates the following files in the output directory:

### Core Output Files

1. **`original_schema.json`** - Schema before compression
   ```json
   {
     "born_in": "The place where someone was born",
     "developed": "Created or improved something",
     "located_in": "The place where something is situated"
   }
   ```

2. **`compressed_schema.json`** - Schema after FAISS compression
   ```json
   {
     "born_in": "The place where someone was born",
     "developed": "Created or improved something"
   }
   ```

3. **`extracted_triplets.json`** - All extracted triplets per sample
   ```json
   [
     [
       ["Albert_Einstein", "born_in", "Ulm_Germany"],
       ["Albert_Einstein", "developed", "theory_of_relativity"]
     ]
   ]
   ```

4. **`contexts.json`** - Original text contexts
   ```json
   [
     "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity."
   ]
   ```

### Metadata Files

5. **`pipeline_metadata.json`** - Processing statistics
   ```json
   {
     "dataset_info": {
       "dataset_type": "json",
       "total_samples": 3,
       "samples_processed": 3
     },
     "pipeline_results": {
       "original_schema_size": 15,
       "compressed_schema_size": 10,
       "compression_ratio": 0.667,
       "total_triplets_extracted": 45
     }
   }
   ```

6. **`summary_report.txt`** - Human-readable summary
   ```
   Schema Definition and Compression Pipeline Report
   ==================================================
   
   Dataset Type: json
   Input File: data/json/sample_test.json
   Samples Processed: 3
   Total Triplets Extracted: 6
   
   Original Schema Size: 15 relations
   Compressed Schema Size: 10 relations
   Compression Ratio: 0.667
   
   Original Schema Relations:
    1. born_in: The place where someone was born
    2. developed: Created or improved something
   ...
   ```

## Compression Methods

### FAISS Max Size (`faiss_max_size`)

Compresses schema to exact number of relations:

```bash
--compression-method faiss_max_size --max-size 15
```

- **Use case**: When you need exactly N relations
- **Parameter**: `--max-size` (required)
- **Result**: Schema with ≤ max_size relations

### FAISS Ratio (`faiss_ratio`)

Compresses schema by percentage:

```bash
--compression-method faiss_ratio --compression-ratio 0.6
```

- **Use case**: When you want to reduce by percentage
- **Parameter**: `--compression-ratio` (required, 0.0-1.0)
- **Result**: Schema with ~(ratio × original_size) relations

### FAISS Similarity (`faiss_similarity`)

Groups relations by semantic similarity:

```bash
--compression-method faiss_similarity --similarity-threshold 0.8
```

- **Use case**: When you want semantic grouping
- **Parameter**: `--similarity-threshold` (optional, default 0.8)
- **Result**: Schema with semantically distinct relations

## Performance Tips

1. **Use `--max-samples`** for testing with large datasets:
   ```bash
   --max-samples 100  # Process only first 100 samples
   ```

2. **Disable synonyms** for faster processing:
   ```bash
   --no-synonyms  # Skip synonym extraction
   ```

3. **Choose appropriate compression method**:
   - `faiss_max_size`: When you have size constraints
   - `faiss_ratio`: When you want proportional reduction
   - `faiss_similarity`: When semantic quality matters most

## Troubleshooting

### Common Issues

1. **Empty context error**: Check that your JSON has proper `context` fields
2. **No triplets extracted**: The model might not be generating proper triplets - check the raw data format
3. **Memory issues**: Use `--max-samples` to limit processing
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging by modifying the script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

This will show detailed processing information including:
- Raw context extraction
- Triplet generation attempts
- Schema compression steps

## Integration with Existing Pipelines

The script can be integrated into larger workflows:

1. **Batch Processing**: Process multiple files in a loop
2. **Configuration-Driven**: Use YAML configs to define parameters
3. **Result Analysis**: Parse `pipeline_metadata.json` for statistics
4. **Quality Assessment**: Compare original vs compressed schemas

## Example Workflow

```bash
# 1. Process dataset
python scripts/run_schema_pipeline.py \
    --input data/json/my_dataset.json \
    --dataset json \
    --output results/my_experiment \
    --compression-method faiss_max_size \
    --max-size 25

# 2. Check results
cat results/my_experiment/summary_report.txt

# 3. Use compressed schema
python my_downstream_task.py \
    --schema results/my_experiment/compressed_schema.json
```

This pipeline provides a complete solution for schema extraction and compression, suitable for knowledge graph construction and question-answering systems.
