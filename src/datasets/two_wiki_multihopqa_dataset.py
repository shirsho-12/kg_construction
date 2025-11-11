"""
2WikiMultiHopQA dataset class.
Handles the specific format of 2WikiMultiHopQA JSON data.
"""

from typing import Dict, List, Any
from pathlib import Path
import logging

from .base_json_dataset import BaseJSONDataset

logger = logging.getLogger(__name__)


class TwoWikiMultiHopQADataset(BaseJSONDataset):
    """
    Dataset for 2WikiMultiHopQA data.

    Expected format:
    - _id: Primary key
    - type: Question type (compositional, argumentative, etc.)
    - question: Question to be answered by QA model
    - context: List of [entity, [sentences]] pairs
    - evidences: List of [entity, relation, entity] triples for evaluation
    - answer: Answer to the question
    """

    def __init__(self, data_path: Path, task_type: str = "graph_construction"):
        super().__init__(data_path, task_type)
        self.dataset_name = "2WikiMultiHopQA"

    def _format_for_graph_construction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for graph construction task."""
        base_fields = self._get_base_fields(sample)

        # Process context into entity: sentences format
        entity_sentences = self._process_context(sample.get("context", []))

        return {
            "_id": base_fields["_id"],
            "type": base_fields["type"],
            "context": entity_sentences,  # entity: sentences format
            "question": base_fields["question"],
            "evidences": sample.get("evidences", []),  # For evaluation
            "answer": base_fields["answer"],
        }

    def _format_for_qa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for QA task."""
        base_fields = self._get_base_fields(sample)

        # For QA, combine context into a single string with entity labels
        context_parts = []
        for entity_info in sample.get("context", []):
            if len(entity_info) >= 2:
                entity_name = entity_info[0]
                sentences = entity_info[1]
                if isinstance(sentences, list):
                    context_text = " ".join(sentences)
                    context_parts.append(f"{entity_name}: {context_text}")

        return {
            "_id": base_fields["_id"],
            "type": base_fields["type"],
            "context": " ".join(context_parts),
            "question": base_fields["question"],
            "answer": base_fields["answer"],
        }

    def get_evidence_triplets(self, sample_idx: int) -> List[List[str]]:
        """
        Extract evidence triplets for evaluation.

        Args:
            sample_idx: Index of the sample

        Returns:
            List of [entity, relation, entity] triplets
        """
        sample = self.data[sample_idx]
        return sample.get("evidences", [])

    def get_question_types(self) -> List[str]:
        """Get all unique question types in the dataset."""
        return list(set(sample.get("type", "") for sample in self.data))

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        question_types = self.get_question_types()
        avg_context_entities = sum(
            len(sample.get("context", [])) for sample in self.data
        ) / len(self.data)
        avg_evidences = sum(
            len(sample.get("evidences", [])) for sample in self.data
        ) / len(self.data)

        return {
            "total_samples": len(self.data),
            "question_types": question_types,
            "num_question_types": len(question_types),
            "avg_context_entities_per_sample": avg_context_entities,
            "avg_evidences_per_sample": avg_evidences,
        }
