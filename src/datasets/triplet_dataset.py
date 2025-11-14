from torch.utils.data import Dataset
import json
from utils.log import logger


class TripletDataset(Dataset):
    """Dataset class for loading triplet data."""

    def __init__(self, triplet_file: str):
        """
        Initialize triplet dataset.

        Args:
            triplets: List of triplet samples, each sample is a dict with 'head', 'relation', 'tail' keys.
        """
        self.triplets = json.load(open(triplet_file, "r", encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: str):
        triplets = self.triplets[idx]
        res = []
        for triplet in triplets:
            if len(triplet) != 3:
                logger.warning(f"Invalid triplet format: {triplet}")
                continue
            head, relation, tail = triplet
            res.append((head, relation, tail))
        return res
