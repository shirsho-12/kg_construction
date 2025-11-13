# Schema Pipeline Script

This directory contains the main schema processing pipeline script.

## Files

- **`run_schema_pipeline.py`** - Main pipeline script for schema definition and compression

## Quick Start

```bash
# Basic usage
python scripts/run_schema_pipeline.py \
    --input data/json/sample_test.json \
    --dataset json \
    --output output/test_run \
    --compression-method faiss_max_size \
    --max-size 10

# With options
python scripts/run_schema_pipeline.py \
    --input data/json/hotpotqa.json \
    --dataset hotpotqa \
    --output output/hotpotqa_run \
    --compression-method faiss_ratio \
    --compression-ratio 0.6 \
    --max-samples 100 \
    --no-synonyms
```

## What it does

1. **Loads dataset** using appropriate Dataset class
2. **Extracts triplets** from text using OIE
3. **Defines schema** from triplets  
4. **Compresses schema** using FAISS methods
5. **Saves results** including schemas, triplets, and metadata

## Output

Creates directory with:
- `original_schema.json` - Schema before compression
- `compressed_schema.json` - Schema after compression
- `extracted_triplets.json` - All extracted triplets
- `contexts.json` - Original text contexts
- `pipeline_metadata.json` - Processing statistics
- `summary_report.txt` - Human-readable summary

## Documentation

See `docs/schema_pipeline_usage.md` for detailed usage guide.

## Requirements

- All dependencies from `requirements.txt`
- Access to the Ministral model for text processing
- Sufficient memory for FAISS operations
