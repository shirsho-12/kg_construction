"""
HotpotQA dataset class.
Handles the specific format of HotpotQA JSON data.
"""

from typing import Dict, List, Any
from pathlib import Path
import logging

from .base_json_dataset import BaseJSONDataset

logger = logging.getLogger(__name__)


class HotpotQADataset(BaseJSONDataset):
    """
    Dataset for HotpotQA data.

    Expected format:
    - _id: Primary key
    - question: Question to be answered
    - answer: Answer to the question
    - context: List of [entity, [sentences]] pairs
    - type: Question type
    - level: Difficulty level (easy, medium, hard, etc.)
    """

    def __init__(self, data_path: Path, task_type: str = "graph_construction"):
        super().__init__(data_path, task_type)
        self.dataset_name = "HotpotQA"

    def _format_for_graph_construction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for graph construction task."""
        base_fields = self._get_base_fields(sample)

        # Process context into entity: sentences format
        entity_sentences = self._process_context(sample.get("context", []))

        return {
            "_id": base_fields["_id"],
            "type": base_fields["type"],
            "level": sample.get("level", ""),
            "context": entity_sentences,  # entity: sentences format
            "question": base_fields["question"],
            "answer": base_fields["answer"],
            "evidences": [],  # HotpotQA doesn't have explicit evidences in this format
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
            "level": sample.get("level", ""),
            "context": " ".join(context_parts),
            "question": base_fields["question"],
            "answer": base_fields["answer"],
        }

    def get_question_types(self) -> List[str]:
        """Get all unique question types in the dataset."""
        return list(set(sample.get("type", "") for sample in self.data))

    def get_difficulty_levels(self) -> List[str]:
        """Get all unique difficulty levels in the dataset."""
        return list(set(sample.get("level", "") for sample in self.data))

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        question_types = self.get_question_types()
        difficulty_levels = self.get_difficulty_levels()
        try:
            avg_context_entities = sum(
                len(sample.get("context", [])) for sample in self.data
            ) / len(self.data)
        except ZeroDivisionError:
            avg_context_entities = 0

        return {
            "total_samples": len(self.data),
            "question_types": question_types,
            "num_question_types": len(question_types),
            "difficulty_levels": difficulty_levels,
            "num_difficulty_levels": len(difficulty_levels),
            "avg_context_entities_per_sample": avg_context_entities,
        }

    def get_samples_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get all samples of a specific difficulty level."""
        return [sample for sample in self.data if sample.get("level") == level]

    def get_samples_by_type(self, question_type: str) -> List[Dict[str, Any]]:
        """Get all samples of a specific question type."""
        return [sample for sample in self.data if sample.get("type") == question_type]
