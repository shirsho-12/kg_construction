"""
Base JSON dataset class for structured data processing.
Provides common functionality for different JSON dataset formats.
"""

from torch.utils.data import Dataset
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseJSONDataset(Dataset):
    """Base dataset for loading structured JSON data."""

    def __init__(self, data_path: Path, task_type: str = "graph_construction"):
        """
        Initialize base JSON dataset.

        Args:
            data_path: Path to JSON file
            task_type: Either "graph_construction" or "qa"
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Handle both single JSON object and array of objects
        if isinstance(self.data, dict):
            self.data = [self.data]

        self.task_type = task_type
        self.dataset_name = self.__class__.__name__
        logger.info(
            f"Loaded {len(self.data)} samples from {self.dataset_name} for {task_type} task"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        if self.task_type == "graph_construction":
            return self._format_for_graph_construction(sample)
        elif self.task_type == "qa":
            return self._format_for_qa(sample)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _get_base_fields(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common fields from sample."""
        return {
            "_id": sample.get("_id"),
            "type": sample.get("type", ""),
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
        }

    def _process_context(self, context: List[List[Any]]) -> Dict[str, str]:
        """
        Process context field from list of [entity, [sentences]] format.

        Args:
            context: List where each item is [entity_name, [sentence1, sentence2, ...]]

        Returns:
            Dictionary mapping entity names to combined sentences
        """
        entity_sentences = {}
        for entity_info in context:
            if len(entity_info) >= 2:
                entity_name = entity_info[0]
                sentences = entity_info[1]
                if isinstance(sentences, list):
                    entity_sentences[entity_name] = " ".join(sentences)
                else:
                    entity_sentences[entity_name] = str(sentences)
        return entity_sentences

    def _format_for_graph_construction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for graph construction task. To be implemented by subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _format_for_graph_construction"
        )

    def _format_for_qa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for QA task. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _format_for_qa")

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get sample by its _id field."""
        for sample in self.data:
            if sample.get("_id") == sample_id:
                return sample
        return None

    def get_all_ids(self) -> List[str]:
        """Get all _id values in the dataset."""
        return [sample.get("_id", f"sample_{i}") for i, sample in enumerate(self.data)]
