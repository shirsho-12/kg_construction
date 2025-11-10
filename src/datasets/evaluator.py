"""
Graph construction evaluator for comparing predicted triplets with ground truth.
"""
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


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
        """
        Evaluate predicted triplets against ground truth evidences.
        
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
        # TODO: Add semantic matching using encoder if available
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
        """
        Check if two triplets match. For now using exact matching.
        TODO: Implement semantic matching using encoder.
        """
        # Case-insensitive matching for entities and relations
        e1_match = triplet1[0].lower().strip() == triplet2[0].lower().strip()
        r_match = triplet1[1].lower().strip() == triplet2[1].lower().strip()
        e2_match = triplet1[2].lower().strip() == triplet2[2].lower().strip()

        return e1_match and r_match and e2_match

    def evaluate_dataset(
        self,
        dataset,
        predicted_triplets_list: List[List[Tuple[str, str, str]]],
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            dataset: JSONDataset with evidences
            predicted_triplets_list: List of predicted triplets for each sample
            
        Returns:
            Overall evaluation metrics
        """
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
