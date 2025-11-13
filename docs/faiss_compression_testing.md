# FAISS Schema Compression Testing Guide

This document describes the testing framework for FAISS-based schema compression with maximum size and compression ratio options.

## Overview

The testing system has been updated to focus exclusively on FAISS compression methods, removing the previous HDBSCAN and Agglomerative clustering approaches. The new system provides precise control over schema compression through two main approaches:

1. **Maximum Size Compression** (`faiss_max_size`) - Compress to exact number of relations
2. **Compression Ratio** (`faiss_ratio`) - Compress by percentage of original size

## Configuration Files

### Base Configuration (`config_templates/base_config.yaml`)

The base configuration now includes FAISS compression settings:

```yaml
pipeline_defaults:
  compression_method: faiss_max_size
  max_schema_size: 20
  compression_ratio: null
  compress_if_more_than: 30

faiss_compression_tests:
  max_size_tests:
    - max_size: 10
      description: "Small schema (10 relations)"
    - max_size: 20  
      description: "Medium schema (20 relations)"
    - max_size: 50
      description: "Large schema (50 relations)"
  
  ratio_tests:
    - ratio: 0.3
      description: "Aggressive compression (30% of original)"
    - ratio: 0.5
      description: "Moderate compression (50% of original)"
    - ratio: 0.7
      description: "Light compression (70% of original)"
```

### Pipeline-Specific Configurations

#### FAISS Max Size Pipeline (`faiss_max_size_pipeline.yaml`)
- Tests compression to specific maximum number of relations
- Includes multiple test configurations for different target sizes

#### FAISS Ratio Pipeline (`faiss_ratio_pipeline.yaml`)  
- Tests compression by percentage of original schema size
- Includes configurations for 30%, 50%, and 70% compression ratios

#### Dataset-Specific Configurations
- **2WikiMultiHopQA**: Uses `faiss_max_size` with 25 relations maximum
- **HotpotQA**: Uses `faiss_ratio` with 60% compression ratio

## Test Files

### Core Compression Tests (`tests/test_schema_compression.py`)

Tests the fundamental FAISS compression functionality:

1. **`test_faiss_max_size_compression()`**
   - Tests compression to specific maximum sizes (5, 10, 15, 20 relations)
   - Verifies target sizes are achieved
   - Reports compression statistics

2. **`test_faiss_ratio_compression()`**
   - Tests compression ratios (30%, 50%, 70%, 80%)
   - Validates actual ratios match targets within tolerance
   - Measures compression effectiveness

3. **`test_compression_quality()`**
   - Tests semantic quality preservation during compression
   - Uses schema with clear semantic groups
   - Verifies important relation types are preserved

### Configuration-Based Tests (`tests/test_faiss_config_compression.py`)

Tests compression using actual configuration files:

1. **`test_config_based_compression()`**
   - Loads test configurations from `base_config.yaml`
   - Tests all max size and ratio configurations
   - Validates results against configuration expectations

2. **`test_pipeline_configs()`**
   - Tests all pipeline-specific configuration files
   - Validates compression settings work as intended
   - Ensures configuration compatibility

## Running Tests

### Basic Compression Tests
```bash
python tests/test_schema_compression.py
```

### Configuration-Based Tests  
```bash
python tests/test_faiss_config_compression.py
```

### Example Usage Test
```bash
python examples/faiss_schema_compression_example.py
```

## Test Schema Design

The tests use carefully designed schemas with semantic groups:

- **Location Relations**: "is located in", "is situated in", "is found in", "resides in"
- **Employment Relations**: "works for", "is employed by", "has job at"  
- **Creation Relations**: "created by", "authored by", "written by", "composed by"
- **Achievement Relations**: "won", "received", "awarded", "earned"
- **Birth/Origin Relations**: "born in", "birth place", "originated from"

This design allows testing of:
- Semantic similarity detection
- Appropriate relation merging
- Quality preservation during compression

## Expected Results

### Maximum Size Compression
- **Target**: Exact number of relations specified
- **Tolerance**: Should achieve target or be within 1-2 relations if insufficient similar relations exist
- **Quality**: Should preserve semantic diversity

### Ratio Compression  
- **Target**: Percentage of original schema size
- **Tolerance**: Within 10% of target ratio
- **Quality**: Should maintain representative relations from each semantic group

## Validation Criteria

Tests validate:

1. **Size Accuracy**: Compressed schemas meet size targets
2. **Ratio Accuracy**: Actual compression ratios match targets within tolerance
3. **Semantic Quality**: Important relation types are preserved
4. **Configuration Compatibility**: All config files work correctly
5. **Error Handling**: Graceful handling of edge cases

## Performance Metrics

Each test reports:
- Original schema size
- Target size/ratio
- Actual compressed size
- Compression percentage
- Relations removed/preserved
- Success/failure status

This comprehensive testing framework ensures FAISS compression works reliably across different scenarios and configuration settings.
