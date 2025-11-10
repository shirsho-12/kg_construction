from torch.utils.data import Dataset
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, data_path, encoder):
        with open(data_path, "r") as f:
            self.data = f.readlines()
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def encode(self, text):
        return self.encoder(text)


class JSONDataset(Dataset):
    """Dataset for loading structured JSON data for graph construction and QA."""

    def __init__(self, data_path: Path, task_type: str = "graph_construction"):
        """
        Initialize JSON dataset.

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
        logger.info(f"Loaded {len(self.data)} samples for {task_type} task")

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

    def _format_for_graph_construction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for graph construction task."""
        # Convert nested context to entity: sentences format
        entity_sentences = {}
        for entity_info in sample.get("context", []):
            if len(entity_info) >= 2:
                entity_name = entity_info[0]
                sentences = entity_info[1]
                entity_sentences[entity_name] = " ".join(sentences)

        return {
            "id": sample.get("_id"),
            "type": sample.get("type"),
            "context": entity_sentences,  # entity: sentences format
            "question": sample.get("question", ""),
            "evidences": sample.get("evidences", []),  # For evaluation
            "answer": sample.get("answer", ""),
        }

    def _format_for_qa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format sample for QA task."""
        # Convert nested context to plain text for QA
        context_text = []
        for entity_info in sample.get("context", []):
            if len(entity_info) >= 2:
                entity_name = entity_info[0]
                sentences = entity_info[1]
                context_text.append(f"{entity_name}: {' '.join(sentences)}")

        return {
            "id": sample.get("_id"),
            "type": sample.get("type"),
            "question": sample.get("question", ""),
            "context": "\n".join(context_text),  # Plain text context
            "answer": sample.get("answer", ""),
            "evidences": sample.get("evidences", []),  # For evaluation
        }

    def get_all_evidences(self) -> List[List[Tuple[str, str, str]]]:
        """Get all evidences for evaluation."""
        return [sample.get("evidences", []) for sample in self.data]

    def get_types(self) -> List[str]:
        """Get all sample types."""
        return [sample.get("type", "") for sample in self.data]


class GraphConstructionEvaluator:
    """Evaluator for graph construction using evidences."""

    def __init__(self, encoder=None):
        self.encoder = encoder
        logger.info("Initialized GraphConstructionEvaluator")

    def evaluate_triplets(
        self,
        predicted_triplets: List[Tuple[str, str, str]],
        ground_truth_evidences: List[Tuple[str, str, str]],
    ) -> Dict[str, float]:
        """Evaluate predicted triplets against ground truth evidences.
        Args:
            predicted_triplets: List of (entity, relation, entity) triplets
            ground_truth_evidences: List of (entity, relation, entity) evidences
        Returns:
                Dictionary with precision, recall, f1 scores
        """
        if not ground_truth_evidences:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if not predicted_triplets:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            # Match triplets using exact matching for now
            #  TODO: Add semantic matching using encoder if available
        matches = 0
        for pred_triplet in predicted_triplets:
            for gt_triplet in ground_truth_evidences:
                if self._triplets_match(pred_triplet, gt_triplet):
                    matches += 1
                    break
        precision = matches / len(predicted_triplets) if predicted_triplets else 0.0
        recall = (
            matches / len(ground_truth_evidences) if ground_truth_evidences else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": len(predicted_triplets),
            "ground_truth_count": len(ground_truth_evidences),
            "matches": matches,
        }

    def _triplets_match(
        self, triplet1: Tuple[str, str, str], triplet2: Tuple[str, str, str]
    ) -> bool:
        """Check if two triplets match. For now using exact matching.
        TODO: Implement semantic matching using encoder."""
        # Case-insensitive matching for entities and relations
        e1_match = triplet1[0].lower().strip() == triplet2[0].lower().strip()
        r_match = triplet1[1].lower().strip() == triplet2[1].lower().strip()
        e2_match = triplet1[2].lower().strip() == triplet2[2].lower().strip()

        return e1_match and r_match and e2_match

    def evaluate_dataset(
        self,
        dataset: JSONDataset,
        predicted_triplets_list: List[List[Tuple[str, str, str]]],
    ) -> Dict[str, Any]:
        """Evaluate entire dataset.
        Args:
            dataset: JSONDataset with evidences
            predicted_triplets_list: List of predicted triplets for each sample
        Returns:
            Overall evaluation metrics"""
        all_scores = []
        type_scores = {}
        for i, sample in enumerate(dataset):
            if i < len(predicted_triplets_list):
                predicted = predicted_triplets_list[i]
                ground_truth = sample.get("evidences", [])
                sample_type = sample.get("type", "unknown")

                scores = self.evaluate_triplets(predicted, ground_truth)
                all_scores.append(scores)

                # Track scores by type
                if sample_type not in type_scores:
                    type_scores[sample_type] = []
                type_scores[sample_type].append(scores)

        # Calculate overall averages
        overall_metrics = {
            "overall_precision": (
                sum(s["precision"] for s in all_scores) / len(all_scores)
                if all_scores
                else 0.0
            ),
            "overall_recall": (
                sum(s["recall"] for s in all_scores) / len(all_scores)
                if all_scores
                else 0.0
            ),
            "overall_f1": (
                sum(s["f1"] for s in all_scores) / len(all_scores)
                if all_scores
                else 0.0
            ),
            "total_samples": len(all_scores),
        }

        # Calculate per-type metrics
        for type_name, scores in type_scores.items():
            overall_metrics[f"{type_name}_precision"] = (
                sum(s["precision"] for s in scores) / len(scores) if scores else 0.0
            )
            overall_metrics[f"{type_name}_recall"] = (
                sum(s["recall"] for s in scores) / len(scores) if scores else 0.0
            )
            overall_metrics[f"{type_name}_f1"] = (
                sum(s["f1"] for s in scores) / len(scores) if scores else 0.0
            )

        return overall_metrics


if __name__ == "__main__":
    from encoder import Encoder
    from config import (
        BASE_ENCODER_MODEL,
        EXAMPLE_DATA_PATH_TEXT,
        EXAMPLE_DATA_PATH_JSON,
    )

    # Test original text dataset
    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    dataset = TextDataset(
        data_path=EXAMPLE_DATA_PATH_TEXT,
        encoder=encoder,
    )
    for i in range(min(3, len(dataset))):
        encoded = dataset[i]
        print(f"Text sample {i}: {encoded}")

    print("\n" + "=" * 50 + "\n")

    # Test JSON dataset for graph construction
    json_dataset_gc = JSONDataset(
        data_path=EXAMPLE_DATA_PATH_JSON, task_type="graph_construction"
    )

    print(f"JSON Graph Construction Dataset: {len(json_dataset_gc)} samples")
    for i in range(min(2, len(json_dataset_gc))):
        sample = json_dataset_gc[i]
        print(f"Sample {i} (type: {sample['type']}):")
        print(f"  Context entities: {list(sample['context'].keys())}")
        print(f"  Question: {sample['question']}")
        print(f"  Evidences: {sample['evidences']}")
        print()

    print("=" * 50 + "\n")

    # Test JSON dataset for QA
    json_dataset_qa = JSONDataset(data_path=EXAMPLE_DATA_PATH_JSON, task_type="qa")

    print(f"JSON QA Dataset: {len(json_dataset_qa)} samples")
    for i in range(min(2, len(json_dataset_qa))):
        sample = json_dataset_qa[i]
        print(f"Sample {i} (type: {sample['type']}):")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Context preview: {sample['context'][:100]}...")
        print()

    # Test evaluator
    print("=" * 50 + "\n")
    evaluator = GraphConstructionEvaluator()

    # Example evaluation
    predicted = [("Lothair II", "mother", "Ermengarde of Tours")]
    ground_truth = [("Lothair II", "mother", "Ermengarde of Tours")]

    scores = evaluator.evaluate_triplets(predicted, ground_truth)
    print(f"Evaluation scores: {scores}")


class CombinedTextDataset(Dataset):
    """Dataset that yields combined text per JSON sample for OIE extraction."""

    def __init__(self, base_ds: JSONDataset):
        """
        Initialize with a JSONDataset.

        Args:
            base_ds: JSONDataset instance with graph_construction samples
        """
        self.base = base_ds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> str:
        sample = self.base[idx]
        entity_context = sample["context"]
        # Combine all entity sentences into one text
        combined_text = " ".join(entity_context.values())
        return combined_text
