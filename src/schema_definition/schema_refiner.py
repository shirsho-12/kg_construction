#!/usr/bin/env python3
"""
Schema Definition Refinement Script

This script allows running schema definition without re-running entity extraction.
It works with both text and JSON pipelines, using previously generated triplet files
from the output directory.

"""

from typing import List, Dict, Tuple, Optional
from schema_definition import FaissSchemaCompressor
from utils import logger


class SchemaRefiner:
    """Schema definition refinement without entity extraction."""

    def __init__(
        self,
        faiss_compressor: FaissSchemaCompressor,
        compression_method: str = "faiss_similarity",
        merge_strategy: str = "llm",
        compression_ratio: float = 0.8,
        compress_if_more_than: int = 30,
        similarity_threshold: float = 0.75,
        max_schema_size: Optional[int] = None,
    ):
        """
        Initialize schema refiner.

        Args:
            compression_method: Schema compression method
            merge_strategy: Strategy for merging relations
            compression_threshold: Threshold for compression
            compress_if_more_than: Minimum relations before compression
        """
        self.compression_method = compression_method
        self.merge_strategy = merge_strategy
        self.compression_ratio = compression_ratio
        self.compress_if_more_than = compress_if_more_than
        self.max_schema_size = max_schema_size
        self.similarity_threshold = similarity_threshold
        self.faiss_compressor = faiss_compressor

        # Initialize components
        logger.info("Initialized Schema Refiner")

    def refine_schema(
        self,
        original_schema: Dict,
    ) -> Tuple[Dict, Dict]:
        """
        Refine schema using triplets and input text.

        Args:
            triplets: List of triplet tuples
            original_schema: Original schema definition
        Returns:
            Refined schema definition
        """
        if not original_schema:
            logger.warning("Empty schema provided; skipping compression.")
            return {}, {}
        if len(original_schema) == 1:
            logger.info("Only one relation type; skipping compression.")
            return original_schema, {}

        logger.info("Running schema refinement")
        try:
            original_count = len(original_schema)
            logger.info(
                f"Starting compression with {original_count} relations using '{self.compression_method}' method (threshold={self.compression_ratio})"
            )
            if (
                self.compression_method == "faiss_max_size"
                and self.max_schema_size is not None
            ):
                compressed_schema, mapping = self.faiss_compressor.compress_by_max_size(
                    schema=original_schema,
                    max_size=self.max_schema_size,
                    merge_strategy=self.merge_strategy,
                )
            elif (
                self.compression_method == "faiss_ratio"
                and self.compression_ratio is not None
            ):
                compressed_schema, mapping = self.faiss_compressor.compress_by_ratio(
                    schema=original_schema,
                    compression_ratio=self.compression_ratio,
                    merge_strategy=self.merge_strategy,
                )
            elif self.compression_method == "faiss_similarity":
                compressed_schema, mapping = (
                    self.faiss_compressor.compress_by_similarity_groups(
                        schema=original_schema,
                        similarity_threshold=self.similarity_threshold,
                        merge_strategy=self.merge_strategy,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown compression method: {self.compression_method}. Available methods: 'faiss_max_size', 'faiss_ratio', 'faiss_similarity'"
                )

            compressed_count = len(compressed_schema)
            compression_ratio = (
                compressed_count / original_count if original_count > 0 else 1.0
            )
            logger.info(
                f"Compression completed: {original_count} -> {compressed_count} relations "
                f"(ratio: {compression_ratio:.3f})"
            )
            if compressed_schema:
                logger.info(f"Compressed to {len(compressed_schema)} relations")
            # Log compression details
            if compressed_count < original_count:
                logger.info(
                    f"Successfully compressed schema by {original_count - compressed_count} relations"
                )
                compressed_relations = list(compressed_schema.keys())
                original_relations = list(original_schema.keys())
                removed_relations = set(original_relations) - set(compressed_relations)
                logger.debug(
                    f"Relations removed by compression: {list(removed_relations)}"
                )
            else:
                logger.info("No compression occurred - relations were too dissimilar")

        except Exception as e:
            logger.error(f"Error during schema compression: {e}")
        return compressed_schema, mapping

    def swap_relations_to_compressed(
        self, oie_triplets: list, compression_mapping: dict
    ) -> list:
        swapped_triplets = []
        for triplet in oie_triplets:
            subj, rel, obj = triplet
            new_rel = compression_mapping.get(rel, rel)
            swapped_triplets.append((subj, new_rel, obj))
        return swapped_triplets
