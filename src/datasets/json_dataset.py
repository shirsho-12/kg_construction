"""
JSON dataset for structured data processing with multiple extraction modes.
Supports base, chunking, and sentence-level extraction.
"""

from torch.utils.data import Dataset
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        # For QA, we might want full context as a single string
        context_parts = []
        for entity_info in sample.get("context", []):
            if len(entity_info) >= 2:
                entity_name = entity_info[0]
                sentences = entity_info[1]
                context_parts.append(f"{entity_name}: {' '.join(sentences)}")

        return {
            "id": sample.get("_id"),
            "type": sample.get("type"),
            "context": " ".join(context_parts),
            "question": sample.get("question", ""),
            "answer": sample.get("answer", ""),
        }


class CombinedTextDataset(Dataset):
    """
    Dataset that yields text per JSON sample for OIE extraction.
    Supports multiple modes: base, chunking, and sentence-level extraction.
    """

    def __init__(
        self,
        base_ds: JSONDataset,
        mode: str = "base",
        chunk_size: int = 100,
    ):
        """
        Initialize with a JSONDataset and extraction mode.

        Args:
            base_ds: JSONDataset instance with graph_construction samples
            mode: Extraction mode - "base", "chunking", or "sentence"
            chunk_size: Number of words per chunk (only for chunking mode)
        """
        self.base = base_ds
        self.mode = mode
        self.chunk_size = chunk_size

        # Pre-process and store all text chunks with their sample indices
        self.text_chunks: List[Tuple[int, str]] = []
        self._prepare_chunks()

        logger.info(
            f"CombinedTextDataset initialized with mode={mode}, "
            f"chunk_size={chunk_size}, total_chunks={len(self.text_chunks)}"
        )

    def _prepare_chunks(self):
        """Prepare all text chunks based on the mode."""
        indices = range(len(self.base))
        if self.mode in {"chunking", "sentence"}:
            indices = tqdm(
                indices,
                desc=f"Preparing {self.mode} chunks",
                leave=False,
            )

        for idx in indices:
            sample = self.base[idx]
            entity_context = sample["context"]
            combined_text = " ".join(entity_context.values())

            if self.mode == "base":
                # Single combined text per sample
                self.text_chunks.append((idx, combined_text))

            elif self.mode == "chunking":
                # Split into word chunks
                words = combined_text.split()
                for i in range(0, len(words), self.chunk_size):
                    chunk = " ".join(words[i : i + self.chunk_size])
                    self.text_chunks.append((idx, chunk))

            elif self.mode == "sentence":
                # Split into sentences
                sentences = self._split_sentences(combined_text)
                for sentence in sentences:
                    if sentence.strip():
                        self.text_chunks.append((idx, sentence.strip()))

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Split on . ! ? followed by space or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def __len__(self) -> int:
        return len(self.text_chunks)

    def __getitem__(self, idx: int) -> str:
        """Return text chunk at index."""
        _, text = self.text_chunks[idx]
        return text

    def get_sample_index(self, chunk_idx: int) -> int:
        """Get the original sample index for a chunk index."""
        if chunk_idx >= len(self.text_chunks):
            raise IndexError(
                f"Chunk index {chunk_idx} out of range. Available chunks: 0-{len(self.text_chunks)-1}"
            )
        sample_idx, _ = self.text_chunks[chunk_idx]
        return sample_idx

    def get_chunks_for_sample(self, sample_idx: int) -> List[Tuple[int, str]]:
        """Get all chunk indices and texts for a given sample index."""
        return [
            (i, text)
            for i, (s_idx, text) in enumerate(self.text_chunks)
            if s_idx == sample_idx
        ]
