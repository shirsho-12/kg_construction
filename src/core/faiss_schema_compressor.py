"""
FAISS-based schema compression using similarity search.
Provides two compression modes: maximum schema size and compression ratio.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import faiss
import torch
from utils import logger
from .encoder import Encoder
from evaluation.faiss_index import FaissIndex


class FaissSchemaCompressor:
    """FAISS-based schema compressor using similarity search for relation merging."""
    
    def __init__(self, encoder: Encoder, similarity_threshold: float = 0.8):
        """
        Initialize FAISS schema compressor.
        
        Args:
            encoder: Encoder for generating embeddings
            similarity_threshold: Minimum similarity for merging relations
        """
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized FAISS Schema Compressor")
    
    def compress_by_max_size(
        self, 
        schema: Dict[str, str], 
        max_size: int,
        merge_strategy: str = "most_similar"
    ) -> Dict[str, str]:
        """
        Compress schema to a maximum number of relations using FAISS similarity search.
        
        Args:
            schema: Dictionary of relation -> definition
            max_size: Maximum number of relations in compressed schema
            merge_strategy: Strategy for merging ("most_similar", "representative")
            
        Returns:
            Compressed schema dictionary
        """
        if not schema or len(schema) <= max_size:
            logger.info(f"Schema size {len(schema)} <= max_size {max_size}, no compression needed")
            return schema
            
        logger.info(f"Compressing schema from {len(schema)} to max {max_size} relations")
        
        relations = list(schema.keys())
        definitions = list(schema.values())
        
        # Filter out empty definitions
        valid_items = [(rel, defn) for rel, defn in zip(relations, definitions) 
                      if defn and defn.strip() and defn.strip() not in ["", "N/A", "None", "null"]]
        
        if len(valid_items) <= max_size:
            logger.info(f"After filtering, only {len(valid_items)} valid relations remain")
            return {rel: defn for rel, defn in valid_items}
        
        valid_relations, valid_definitions = zip(*valid_items)
        valid_relations, valid_definitions = list(valid_relations), list(valid_definitions)
        
        # Generate embeddings
        embeddings = self._get_embeddings(valid_definitions)
        
        # Build FAISS index
        faiss_index = FaissIndex(embedding_dim=embeddings.shape[1], normalize=True)
        faiss_index.add(embeddings, list(range(len(valid_relations))))
        
        # Iteratively merge most similar relations until we reach max_size
        current_relations = valid_relations.copy()
        current_definitions = valid_definitions.copy()
        current_embeddings = embeddings.copy()
        merged_indices = set()
        
        while len(current_relations) - len(merged_indices) > max_size:
            # Find the most similar pair
            best_similarity = -1
            best_pair = None
            
            for i in range(len(current_relations)):
                if i in merged_indices:
                    continue
                    
                # Search for most similar relation to current relation
                query_embedding = current_embeddings[i:i+1]
                results = faiss_index.search(query_embedding, top_k=len(current_relations))
                
                for result_idx, similarity in results:
                    if result_idx == i or result_idx in merged_indices:
                        continue
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, result_idx)
                        break
                
                if best_pair:
                    break
            
            if not best_pair:
                logger.warning(f"Cannot find any relations to merge (best similarity: {best_similarity})")
                break
            
            # If we can't find similar enough relations but still need to compress, 
            # lower the threshold temporarily
            if best_similarity < self.similarity_threshold:
                logger.info(f"Lowering similarity threshold to {best_similarity:.3f} to achieve target size")
                # Continue with the best pair found
            
            # Merge the best pair
            idx1, idx2 = best_pair
            merged_relation, merged_definition = self._merge_relations(
                current_relations[idx1], current_definitions[idx1],
                current_relations[idx2], current_definitions[idx2],
                strategy=merge_strategy
            )
            
            # Update the first relation and mark second as merged
            current_relations[idx1] = merged_relation
            current_definitions[idx1] = merged_definition
            merged_indices.add(idx2)
            
            logger.debug(f"Merged '{current_relations[idx2]}' into '{merged_relation}' (similarity: {best_similarity:.3f})")
        
        # Build final compressed schema
        compressed_schema = {}
        for i, (rel, defn) in enumerate(zip(current_relations, current_definitions)):
            if i not in merged_indices:
                compressed_schema[rel] = defn
        
        logger.info(f"Compressed schema from {len(schema)} to {len(compressed_schema)} relations")
        return compressed_schema
    
    def compress_by_ratio(
        self, 
        schema: Dict[str, str], 
        compression_ratio: float,
        merge_strategy: str = "most_similar"
    ) -> Dict[str, str]:
        """
        Compress schema by a specific compression ratio using FAISS similarity search.
        
        Args:
            schema: Dictionary of relation -> definition
            compression_ratio: Target ratio (0.0 to 1.0, where 0.5 means 50% of original size)
            merge_strategy: Strategy for merging ("most_similar", "representative")
            
        Returns:
            Compressed schema dictionary
        """
        if not schema or compression_ratio >= 1.0:
            logger.info(f"Compression ratio {compression_ratio} >= 1.0, no compression needed")
            return schema
        
        target_size = max(1, int(len(schema) * compression_ratio))
        logger.info(f"Compressing schema by ratio {compression_ratio} (target size: {target_size})")
        
        return self.compress_by_max_size(schema, target_size, merge_strategy)
    
    def compress_by_similarity_groups(
        self,
        schema: Dict[str, str],
        similarity_threshold: float = None,
        merge_strategy: str = "most_similar"
    ) -> Dict[str, str]:
        """
        Compress schema by grouping similar relations using FAISS similarity search.
        
        Args:
            schema: Dictionary of relation -> definition
            similarity_threshold: Minimum similarity for grouping (uses instance default if None)
            merge_strategy: Strategy for merging ("most_similar", "representative")
            
        Returns:
            Compressed schema dictionary
        """
        if not schema:
            return schema
            
        threshold = similarity_threshold or self.similarity_threshold
        logger.info(f"Compressing schema by similarity groups (threshold: {threshold})")
        
        relations = list(schema.keys())
        definitions = list(schema.values())
        
        # Filter out empty definitions
        valid_items = [(rel, defn) for rel, defn in zip(relations, definitions) 
                      if defn and defn.strip() and defn.strip() not in ["", "N/A", "None", "null"]]
        
        if len(valid_items) <= 1:
            return {rel: defn for rel, defn in valid_items}
        
        valid_relations, valid_definitions = zip(*valid_items)
        valid_relations, valid_definitions = list(valid_relations), list(valid_definitions)
        
        # Generate embeddings
        embeddings = self._get_embeddings(valid_definitions)
        
        # Build FAISS index
        faiss_index = FaissIndex(embedding_dim=embeddings.shape[1], normalize=True)
        faiss_index.add(embeddings, list(range(len(valid_relations))))
        
        # Find similarity groups
        processed = set()
        compressed_schema = {}
        
        for i, (relation, definition) in enumerate(zip(valid_relations, valid_definitions)):
            if i in processed:
                continue
            
            # Find all similar relations to this one
            query_embedding = embeddings[i:i+1]
            results = faiss_index.search(query_embedding, top_k=len(valid_relations))
            
            similar_indices = []
            for result_idx, similarity in results:
                if result_idx not in processed and similarity >= threshold:
                    similar_indices.append(result_idx)
            
            if len(similar_indices) == 1:
                # No similar relations found, keep as is
                compressed_schema[relation] = definition
                processed.add(i)
            else:
                # Merge similar relations
                group_relations = [valid_relations[idx] for idx in similar_indices]
                group_definitions = [valid_definitions[idx] for idx in similar_indices]
                
                merged_relation, merged_definition = self._merge_relation_group(
                    group_relations, group_definitions, strategy=merge_strategy
                )
                
                compressed_schema[merged_relation] = merged_definition
                processed.update(similar_indices)
                
                logger.debug(f"Merged group of {len(similar_indices)} relations into '{merged_relation}'")
        
        logger.info(f"Compressed schema from {len(schema)} to {len(compressed_schema)} relations")
        return compressed_schema
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text list."""
        embeddings = self.encoder.encode(texts)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings
    
    def _merge_relations(
        self, 
        rel1: str, def1: str, 
        rel2: str, def2: str, 
        strategy: str = "most_similar"
    ) -> Tuple[str, str]:
        """
        Merge two relations based on strategy.
        
        Args:
            rel1, def1: First relation and definition
            rel2, def2: Second relation and definition
            strategy: Merging strategy
            
        Returns:
            Tuple of (merged_relation, merged_definition)
        """
        if strategy == "most_similar":
            # Keep the shorter, more general relation name
            if len(rel1) <= len(rel2):
                return rel1, def1
            else:
                return rel2, def2
        elif strategy == "representative":
            # Choose the relation with longer definition (more informative)
            if len(def1) >= len(def2):
                return rel1, def1
            else:
                return rel2, def2
        else:
            # Default: keep first relation
            return rel1, def1
    
    def _merge_relation_group(
        self, 
        relations: List[str], 
        definitions: List[str], 
        strategy: str = "most_similar"
    ) -> Tuple[str, str]:
        """
        Merge a group of relations based on strategy.
        
        Args:
            relations: List of relation names
            definitions: List of relation definitions
            strategy: Merging strategy
            
        Returns:
            Tuple of (merged_relation, merged_definition)
        """
        if len(relations) == 1:
            return relations[0], definitions[0]
        
        if strategy == "most_similar":
            # Choose the shortest relation name (most general)
            min_idx = min(range(len(relations)), key=lambda i: len(relations[i]))
            return relations[min_idx], definitions[min_idx]
        elif strategy == "representative":
            # Choose the relation with longest definition (most informative)
            max_idx = max(range(len(definitions)), key=lambda i: len(definitions[i]))
            return relations[max_idx], definitions[max_idx]
        else:
            # Default: keep first relation
            return relations[0], definitions[0]
