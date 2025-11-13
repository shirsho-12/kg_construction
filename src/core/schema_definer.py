from typing import Union, List, Optional
from pathlib import Path
import logging
from .encoder import Encoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
import numpy as np
from config import LOGGING_LEVEL
from tqdm import tqdm

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class SchemaDefiner:
    def __init__(
        self,
        model: Encoder,
        schema_prompt_path: Union[str, Path],
        schema_few_shot_examples_path: Union[str, Path],
    ):
        self.model = model
        self.schema_prompt = open(schema_prompt_path).read()
        self.schema_few_shot_examples = open(schema_few_shot_examples_path).read()
        logger.debug(
            "SchemaDefiner initialized with schema prompt path: %s", schema_prompt_path
        )

    def run(self, input_text: str, oie_triplets: list):
        logger.debug("Defining schema for input text: %s", input_text)
        schema_definition_dct = []
        for idx, oie_triplets in enumerate(oie_triplets):
            logger.debug("OIE Triplets for input %d: %s", idx, oie_triplets)
            defined_schema = self.define_schema(input_text, oie_triplets)
            schema_definition_dct.append(defined_schema)
            logger.debug("Defined schema for input %d: %s", idx, defined_schema)
        return schema_definition_dct

    def define_schema(self, input_text: str, oie_triplets: list):
        relations = set()
        for triplet in oie_triplets:
            if isinstance(triplet, list) and len(triplet) == 1:
                triplet = triplet[0]
            if isinstance(triplet, (list, tuple)) and len(triplet) >= 2:
                relations.add(triplet[1])
            elif isinstance(triplet, str):
                new_triplet = triplet.split("#SEP")
                if len(new_triplet) == 3:
                    relations.add(new_triplet[1])
            else:
                logger.warning("Invalid triplet format in define_schema: %s", triplet)
        filled_prompt = self.schema_prompt.format(
            text=input_text,
            few_shot_examples=self.schema_few_shot_examples,
            relations=relations,
            triples=oie_triplets,
        )
        logger.debug("Filled schema prompt: %s", filled_prompt)
        schema = self.model.generate_completion(
            [{"role": "user", "content": filled_prompt}], answer_prefix="OUTPUT::"
        )
        logger.debug("Schema completion received: %s", schema)
        if (
            not schema
            or not schema[0]
            or schema[0].strip() in ["", "\n", "OUTPUT::", "OUTPUT::\n"]
        ):
            logger.warning(
                "Schema generation returned empty; using fallback empty schema."
            )
            return {}
        return self._parse_schema(schema[0])

    def _parse_schema(self, schema_text: str):
        descriptions = schema_text.split("\n")
        schema_dct = {}

        for desc in descriptions:
            if ":" not in desc:
                continue

            colon_idx = desc.index(":")
            relation = desc[:colon_idx].strip()
            definition = desc[colon_idx + 1 :].strip()

            # Remove numbered prefixes like '1.', '2.' from relation and definition
            relation = self._strip_numbered_prefix(relation)
            definition = self._strip_numbered_prefix(definition)

            # Apply filtering rules
            # Skip "{relation}" as a key
            if relation == "{relation}" or relation == "Schema" or not relation:
                continue

            # Remove all instances of \" and \"
            relation = relation.replace('"', "").replace("\\", "")
            definition = definition.replace('"', "").replace("\\", "")

            # Remove instances of curly braces in the text
            relation = relation.replace("{", "").replace("}", "")
            definition = definition.replace("{", "").replace("}", "")

            # Ignore instances where there are no descriptions
            if not definition or definition.strip() in ["", "N/A", "None", "null"]:
                continue

            # Any time the relation is longer than 7 words, ignore
            if len(relation.split()) > 7:
                logger.warning("Skipping relation longer than 7 words: %s", relation)
                continue

            # Only add if both relation and definition are valid after cleaning
            if relation and definition:
                schema_dct[relation] = definition

        logger.info("Parsed schema with %d relations after filtering", len(schema_dct))
        return schema_dct

    def _strip_numbered_prefix(self, text: str) -> str:
        # Remove patterns like '1.', '2.', '1) ', '2) ', etc. from the start
        import re

        text = text.strip()
        text = re.sub(r"^\d+[\.\)]\s*", "", text)
        return text

    def compress_schema(self, schema: dict, threshold=0.8, method="agglomerative"):
        if not schema:
            logger.warning("Empty schema provided; skipping compression.")
            return {}
        if len(schema) == 1:
            logger.info("Only one relation type; skipping compression.")
            return schema

        original_count = len(schema)
        logger.info(
            f"Starting compression with {original_count} relations using {method} method (threshold={threshold})"
        )

        relations = list(schema.keys())
        definitions = list(schema.values())

        # Filter out empty definitions before encoding
        valid_indices = []
        valid_relations = []
        valid_definitions = []

        for i, (rel, definition) in enumerate(zip(relations, definitions)):
            if (
                definition
                and definition.strip()
                and definition.strip() not in ["", "N/A", "None", "null"]
            ):
                valid_indices.append(i)
                valid_relations.append(rel)
                valid_definitions.append(definition)
            else:
                logger.warning(f"Skipping empty definition for relation: {rel}")

        if len(valid_definitions) < 2:
            logger.info(
                f"Only {len(valid_definitions)} valid definitions; skipping compression."
            )
            return {rel: schema[rel] for rel in valid_relations}

        try:
            embeddings = self.model.encode(valid_definitions).cpu().numpy()
            logger.debug(f"Generated embeddings with shape: {embeddings.shape}")

            if method == "agglomerative":
                compressed_schema = self._agglomerative_compress(
                    valid_relations, valid_definitions, embeddings, threshold
                )
            elif method == "hdbscan":
                compressed_schema = self._hdbscan_compress(
                    valid_relations, valid_definitions, embeddings
                )
            else:
                raise ValueError(f"Unknown compression method: {method}")

            compressed_count = len(compressed_schema)
            compression_ratio = (
                compressed_count / original_count if original_count > 0 else 1.0
            )

            logger.info(
                f"Compression completed: {original_count} -> {compressed_count} relations "
                f"(ratio: {compression_ratio:.3f})"
            )

            # Log compression details
            if compressed_count < original_count:
                logger.info(
                    f"Successfully compressed schema by {original_count - compressed_count} relations"
                )
                compressed_relations = list(compressed_schema.keys())
                original_relations = list(schema.keys())
                removed_relations = set(original_relations) - set(compressed_relations)
                logger.debug(
                    f"Relations removed by compression: {list(removed_relations)}"
                )
            else:
                logger.info("No compression occurred - relations were too dissimilar")

            return compressed_schema

        except Exception as e:
            logger.error(f"Compression failed with error: {e}")
            logger.info("Returning original schema due to compression failure")
            return schema

    def _agglomerative_compress(self, relations, definitions, embeddings, threshold):
        logger.debug(f"Running agglomerative clustering with threshold {threshold}")

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
        )
        cluster_labels = clusterer.fit_predict(embeddings)

        logger.debug(f"Cluster labels: {cluster_labels}")
        unique_clusters = set(cluster_labels)
        logger.debug(f"Found {len(unique_clusters)} unique clusters")

        compressed_schema = {}
        clusters_info = {}

        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_relations = [relations[i] for i in cluster_indices]
            cluster_definitions = [definitions[i] for i in cluster_indices]

            # Log cluster information
            clusters_info[int(cluster_id)] = {
                "relations": cluster_relations,
                "count": len(cluster_relations),
            }

            if len(cluster_indices) == 1:
                # Single item cluster, keep as is
                logger.debug(
                    f"Single-item cluster {cluster_id}: {cluster_relations[0]}"
                )
                compressed_schema[cluster_relations[0]] = cluster_definitions[0]
            else:
                # Multiple items in cluster, choose representative
                representative_idx = cluster_indices[0]
                representative_relation = relations[representative_idx]
                representative_definition = definitions[representative_idx]

                logger.debug(
                    f"Multi-item cluster {cluster_id} (size {len(cluster_indices)}): "
                    f"representative={representative_relation}, "
                    f"all_relations={cluster_relations}"
                )

                compressed_schema[representative_relation] = representative_definition

        logger.debug(f"Clustering details: {clusters_info}")
        return compressed_schema

    def _hdbscan_compress(self, relations, definitions, embeddings):
        clusterer = HDBSCAN(
            min_cluster_size=2, metric="cosine", allow_single_cluster=True
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        compressed_schema = {}
        for cluster_id in set(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            representative_idx = cluster_indices[0]
            representative_relation = relations[representative_idx]
            representative_definition = definitions[representative_idx]
            compressed_schema[representative_relation] = representative_definition
        return compressed_schema

    def swap_relations_to_compressed(
        self, oie_triplets: list, original_to_compressed_map: dict
    ) -> list:
        swapped_triplets = []
        for triplet in oie_triplets:
            if isinstance(triplet, list) and len(triplet) == 1:
                triplet = triplet[0]
            if isinstance(triplet, str):
                new_triplet = triplet.split("#SEP")
                if len(new_triplet) == 3:
                    subj, rel, obj = new_triplet
                else:
                    logger.warning("Skipping malformed triplet in swap: %s", triplet)
                    continue
            elif len(triplet) != 3:
                logger.warning("Skipping malformed triplet in swap: %s", triplet)
                continue
            else:
                subj, rel, obj = triplet
            new_rel = original_to_compressed_map.get(rel, rel)
            swapped_triplets.append((subj, new_rel, obj))
        return swapped_triplets

    def save_entities_relations_to_json(
        self,
        oie_triplets: list,
        output_path: Union[str, Path],
    ):
        import json

        results = []
        triplets_text = []  # Also save as text format
        
        for idx, triplet in enumerate(oie_triplets):
            # Handle different triplet formats
            subj, rel, obj = None, None, None
            
            if isinstance(triplet, (list, tuple)) and len(triplet) == 1:
                triplet = triplet[0]
            
            if isinstance(triplet, str):
                new_triplet = triplet.split("#SEP")
                if len(new_triplet) == 3:
                    subj, rel, obj = new_triplet
                    triplets_text.append(triplet)  # Save original string format
                else:
                    logger.warning("Skipping malformed string triplet: %s", triplet)
                    continue
            elif (
                isinstance(triplet, (list, tuple))
                and isinstance(triplet[-1], (list, tuple))
                and isinstance(triplet[-1][0], str)
                and "#SEP" in triplet[-1][0]
            ):
                subj, rel, obj = triplet[-1][0].split("#SEP")
                triplets_text.append(triplet[-1][0])  # Save original string format
            elif len(triplet) == 3 and isinstance(triplet, (list, tuple)):
                subj, rel, obj = triplet
                triplets_text.append(f"{subj}#SEP{rel}#SEP{obj}")  # Create string format
            else:
                logger.warning("Skipping malformed triplet: %s", triplet)
                continue
            
            # Add to results as dictionary format
            if subj and rel and obj:
                results.append({"subject": subj, "relation": rel, "object": obj})

        # Create output data with both formats
        output_data = {
            "triplets": results,
            "triplets_text": triplets_text,
            "count": len(results)
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved %d triples to %s (including text format)", len(results), output_path)

    def save_schema_definitions(self, schemas, output_path: Union[str, Path]):
        """Save schema definitions for each input text."""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schemas, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d schema definitions to %s", len(schemas), output_path)

    def save_compression_outcomes(
        self,
        original_schemas: List[dict],
        compressed_schemas: List[dict],
        compression_method: str,
        output_path: Union[str, Path],
    ):
        """Save compression outcomes showing before/after comparison with detailed metrics."""
        import json

        outcomes = []
        total_original = 0
        total_compressed = 0
        successful_compressions = 0
        
        for i, (orig, comp) in enumerate(zip(original_schemas, compressed_schemas)):
            original_count = len(orig) if orig else 0
            compressed_count = len(comp) if comp else 0
            compression_ratio = compressed_count / original_count if original_count > 0 else 1.0
            reduction_count = original_count - compressed_count
            reduction_percentage = (reduction_count / original_count * 100) if original_count > 0 else 0
            
            # Check if compression actually occurred
            is_compressed = compressed_count < original_count and original_count > 1
            if is_compressed:
                successful_compressions += 1
            
            outcome = {
                "input_index": i,
                "original_relations": list(orig.keys()) if orig else [],
                "compressed_relations": list(comp.keys()) if comp else [],
                "original_count": original_count,
                "compressed_count": compressed_count,
                "compression_ratio": compression_ratio,
                "reduction_count": reduction_count,
                "reduction_percentage": round(reduction_percentage, 2),
                "compression_method": compression_method,
                "compression_successful": is_compressed,
                "removed_relations": list(set(orig.keys()) - set(comp.keys())) if orig and comp else [],
            }
            outcomes.append(outcome)
            
            total_original += original_count
            total_compressed += compressed_count

        # Calculate overall metrics
        overall_compression_ratio = total_compressed / total_original if total_original > 0 else 1.0
        overall_reduction = total_original - total_compressed
        overall_reduction_percentage = (overall_reduction / total_original * 100) if total_original > 0 else 0
        success_rate = (successful_compressions / len(original_schemas) * 100) if original_schemas else 0
        
        summary_metrics = {
            "total_inputs": len(original_schemas),
            "total_original_relations": total_original,
            "total_compressed_relations": total_compressed,
            "overall_compression_ratio": round(overall_compression_ratio, 3),
            "overall_reduction": overall_reduction,
            "overall_reduction_percentage": round(overall_reduction_percentage, 2),
            "successful_compressions": successful_compressions,
            "compression_success_rate": round(success_rate, 2),
            "compression_method": compression_method,
        }

        # Save both detailed outcomes and summary metrics
        output_data = {
            "summary_metrics": summary_metrics,
            "detailed_outcomes": outcomes
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Compression metrics saved to {output_path}")
        logger.info(f"Overall: {total_original} -> {total_compressed} relations "
                   f"({overall_reduction} reduced, {overall_reduction_percentage:.1f}% reduction)")
        logger.info(f"Success rate: {successful_compressions}/{len(original_schemas)} "
                   f"({success_rate:.1f}%)")


if __name__ == "__main__":
    from config import (
        SD_PROMPT_PATH,
        SD_FEW_SHOT_EXAMPLES_PATH,
        BASE_ENCODER_MODEL,
    )
    from datasets import TextDataset
    from torch.utils.data import DataLoader
    from config import EXAMPLE_DATA_PATH_JSON

    model = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    schema_definer = SchemaDefiner(
        model=model,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )
    dataset = TextDataset(
        data_path=EXAMPLE_DATA_PATH_JSON,
        encoder=model,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i, batch in enumerate(dataloader):
        logger.debug(f"Processing batch {i}: {batch}")
        for b in batch:
            oie_triplets = [
                ("Alice", "visited", "Paris"),
                ("Bob", "works_at", "CompanyX"),
            ]
            schema = schema_definer.run(b, [oie_triplets])

            for s in schema:
                for relation, definition in s.items():
                    print(f"Relation: {relation}\nDefinition: {definition}\n")

            compressed_schema = schema_definer.compress_schema(schema[0])
            print(f"Compressed Schema - Agglomerative: {compressed_schema}\n")
            compressed_schema_hdbscan = schema_definer.compress_schema(
                schema[0], method="hdbscan"
            )
            print(f"Compressed Schema - HDBSCAN: {compressed_schema_hdbscan}\n")

            # Build map from original to compressed relations
            original_to_compressed = {}
            if schema[0]:
                if compressed_schema:
                    # Simple heuristic: map in order; replace with proper clustering
                    for orig, comp in zip(schema[0].keys(), compressed_schema.keys()):
                        original_to_compressed[orig] = comp
                else:
                    original_to_compressed = {rel: rel for rel in schema[0].keys()}

            swapped_triplets = schema_definer.swap_relations_to_compressed(
                oie_triplets, original_to_compressed
            )
            print("Swapped triplets:", swapped_triplets)

            # Save to JSON
            output_path = Path.cwd() / "output" / f"batch_{i}_triplets.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            schema_definer.save_entities_relations_to_json(
                swapped_triplets, output_path
            )
