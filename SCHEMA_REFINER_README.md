# Schema Refiner

The Schema Refiner allows you to refine schema definitions without re-running entity extraction. This is useful for iterating on schema definition parameters and testing different compression methods quickly.

## Features

- **Works with both text and JSON pipelines**
- **Three refinement variants**:
  1. Triplets only
  2. Triplets + input text
  3. Triplets + synonyms + input text
- **Flexible file format support** - automatically detects different JSON formats
- **Robust error handling** - checks for file existence and handles format mismatches
- **Progress tracking** - shows progress with tqdm bars
- **Comprehensive output** - saves both schema definitions and compressed schemas

## Usage

### Basic Command

```bash
python src/schema_refiner.py --output-dir OUTPUT_DIR --variant VARIANT
```

### Required Arguments

- `--output-dir`: Path to the output directory containing previous pipeline results
- `--variant`: Schema refinement variant:
  - `triplets_only`: Use only extracted triplets
  - `triplets_text`: Use triplets + input text
  - `triplets_synonyms_text`: Use triplets + synonyms + input text

### Optional Arguments

- `--triplets-file`: Name of triplets file (default: `triplets.json`)
- `--synonyms-file`: Name of synonyms file (default: `synonyms.json`)
- `--input-file`: Name of input file containing texts (default: `results.json`)
- `--compression-method`: Compression method - `agglomerative` or `hdbscan` (default: `agglomerative`)
- `--compression-threshold`: Compression threshold (default: `0.8`)
- `--compress-if-more-than`: Minimum relations before compression (default: `30`)

## Examples

### 1. Triplets Only Refinement

```bash
python src/schema_refiner.py --output-dir output/json/example_data --variant triplets_only
```

This uses only the extracted triplets to generate schema definitions.

### 2. Triplets + Input Text Refinement

```bash
python src/schema_refiner.py --output-dir output/json/example_data --variant triplets_text --compression-threshold 0.7
```

This uses both triplets and the original input text for better context.

### 3. Full Refinement with Synonyms

```bash
python src/schema_refiner.py --output-dir output/json/example_data --variant triplets_synonyms_text --compression-method hdbscan
```

This uses all available data: triplets, synonyms, and input text.

## Input File Formats

The schema refiner automatically detects and handles different JSON formats:

### Triplets Files
- `{"triplets": [[...], [...]]}`
- `{"sample_id": [...], "sample_id2": [...]}`
- `[[...], [...]]`
- `[{"subject": "...", "relation": "...", "object": "..."}, ...]`

### Synonyms Files
- `{"sample_id": {...}, "sample_id2": {...}}`
- `[{}, {}, ...]`

### Input Text Files
- JSON datasets with `context` fields
- Files containing long text strings
- Results files from previous pipeline runs

## Output Files

The schema refiner creates a `schema_refinement` subdirectory in your output directory with the following files:

### Schema Definition Files
- `schema_definitions_{variant}.json`: Original schema definitions
- `compressed_schemas_{variant}.json`: Compressed schema definitions
- `schema_results_{variant}.json`: Combined results with metadata

### Example Output Structure

```json
{
  "sample_id": "sample_0",
  "schema_definition": {
    "is mother of": ["person"],
    "date of death": ["date"]
  },
  "compressed_schema": {
    "parental_relation": ["person"],
    "date_event": ["date"]
  },
  "num_relations": 2,
  "num_compressed_relations": 2
}
```

## File Detection

The schema refiner automatically tries alternative file names if the specified files don't exist:

### Triplets Files
- `triplets.json`
- `triplets_by_id.json`
- `triplets_compressed.json`
- `triplets_compressed_by_id.json`

### Synonyms Files
- `synonyms.json`
- `synonyms_by_id.json`

### Input Files
- `results.json`
- `results_by_id.json`
- `graph_construction_results.json`

## Error Handling

The schema refiner includes robust error handling:

- **File existence checks**: Verifies all required files exist before processing
- **Format validation**: Validates JSON formats and provides helpful error messages
- **Length matching**: Handles mismatches between different data types by using the minimum length
- **Graceful degradation**: Continues processing even if some samples fail
- **Detailed logging**: Provides comprehensive logging for debugging

## Integration with Pipelines

### JSON Pipeline Integration

```python
# After running the main JSON pipeline
from src.schema_refiner import SchemaRefiner

refiner = SchemaRefiner(compression_method="agglomerative", compression_threshold=0.8)
refiner.refine_schema_triplets_synonyms_text(
    triplets_file=Path("output/json/data/triplets_by_id.json"),
    synonyms_file=Path("output/json/data/synonyms_by_id.json"),
    input_file=Path("output/json/data/results_by_id.json"),
    output_dir=Path("output/json/data/schema_refinement")
)
```

### Text Pipeline Integration

```python
# After running the main text pipeline
from src.schema_refiner import SchemaRefiner

refiner = SchemaRefiner()
refiner.refine_schema_triplets_text(
    triplets_file=Path("output/text/data/triplets.json"),
    input_file=Path("output/text/data/results.json"),
    output_dir=Path("output/text/data/schema_refinement")
)
```

## Advanced Usage

### Custom Compression Parameters

```bash
python src/schema_refiner.py \
  --output-dir output/json/data \
  --variant triplets_synonyms_text \
  --compression-method hdbscan \
  --compression-threshold 0.6 \
  --compress-if-more-than 25
```

### Batch Processing

You can create a simple script to test multiple parameter combinations:

```python
from pathlib import Path
import subprocess

output_dir = Path("output/json/data")
variants = ["triplets_only", "triplets_text", "triplets_synonyms_text"]
thresholds = [0.7, 0.8, 0.9]

for variant in variants:
    for threshold in thresholds:
        cmd = [
            "python", "src/schema_refiner.py",
            "--output-dir", str(output_dir),
            "--variant", variant,
            "--compression-threshold", str(threshold)
        ]
        subprocess.run(cmd)
```

## Troubleshooting

### Common Issues

1. **File not found**: Check that the output directory exists and contains the required files
2. **Format errors**: Ensure your JSON files are valid and contain the expected data structure
3. **Length mismatches**: The refiner automatically handles this by using the minimum length
4. **Memory issues**: For large datasets, consider processing in smaller batches

### Debug Mode

Add more verbose logging by modifying the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use appropriate variants**: Start with `triplets_only` for quick iterations
2. **Adjust compression threshold**: Lower thresholds create more compression but may lose specificity
3. **Batch processing**: Process multiple parameter combinations automatically
4. **Monitor memory**: Large datasets with many synonyms can require significant memory

## Related Files

- `src/schema_refiner.py`: Main schema refiner implementation
- `src/schema_refiner_example.py`: Example usage script
- `src/schema_definer.py`: Core schema definition and compression logic
- `src/config.py`: Configuration constants and paths
