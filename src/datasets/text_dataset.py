"""
Text dataset for simple line-by-line text processing.
"""
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple dataset that loads text line by line."""

    def __init__(self, data_path: Path, encoder=None):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to text file
            encoder: Optional encoder for encoding text
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        self.encoder = encoder
        logger.info(f"Loaded {len(self.data)} text lines from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].strip()

    def encode(self, text):
        """Encode text using the encoder if available."""
        if self.encoder:
            return self.encoder(text)
        return text
