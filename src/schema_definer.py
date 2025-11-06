from typing import Optional, Union
from pathlib import Path
import logging

from config import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)
from encoder import Encoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
import numpy as np


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
            relations.add(triplet[1])
        filled_prompt = self.schema_prompt.format(
            text=input_text,
            few_shot_examples=self.schema_few_shot_examples,
            relations=relations,
            triples=oie_triplets,
        )
        logger.debug("Filled schema prompt: %s", filled_prompt)
        schema = self.model.generate_completion(
            [{"role": "user", "content": filled_prompt}], answer_prefix="Schema: "
        )
        logger.debug("Schema completion received: %s", schema)
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
            if relation == "Schema" or not relation:
                continue
            schema_dct[relation] = definition
        return schema_dct

    def compress_schema(self, schema: dict, threshold=0.8, method="agglomerative"):
        relations = list(schema.keys())
        definitions = list(schema.values())
        embeddings = self.model.encode(definitions).cpu().numpy()
        logger.debug("Embeddings for schema definitions: %s", embeddings)
        if method == "agglomerative":
            return self._agglomerative_compress(
                relations, definitions, embeddings, threshold
            )
        elif method == "hdbscan":
            return self._hdbscan_compress(relations, definitions, embeddings)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def _agglomerative_compress(self, relations, definitions, embeddings, threshold):

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
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


if __name__ == "__main__":
    from config import SD_PROMPT_PATH, SD_FEW_SHOT_EXAMPLES_PATH, BASE_ENCODER_MODEL
    from dataset import TextDataset
    from torch.utils.data import DataLoader
    from config import EXAMPLE_DATA_PATH_TEXT

    model = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    schema_definer = SchemaDefiner(
        model=model,
        schema_prompt_path=SD_PROMPT_PATH,
        schema_few_shot_examples_path=SD_FEW_SHOT_EXAMPLES_PATH,
    )
    dataset = TextDataset(
        data_path=EXAMPLE_DATA_PATH_TEXT,
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
