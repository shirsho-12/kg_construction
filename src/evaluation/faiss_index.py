import faiss
import numpy as np
from typing import List, Tuple, Union
import torch


class FaissIndex:
    def __init__(self, embedding_dim: int, normalize: bool = True):
        self.normalize = normalize
        self.d = embedding_dim
        self.index = faiss.IndexFlatIP(
            self.d
        )  # cosine via inner-product on normalized vectors
        self.id_map = []  # list of passage ids (order matches vectors in index)

    def add(self, vectors: Union[np.ndarray, torch.Tensor], ids: List[int]):
        # vectors shape (n, d)
        # Convert torch tensor to numpy if needed
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()

        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        self.index.add(vectors.astype("float32"))
        self.id_map.extend(ids)

    def search(self, query_vec: Union[np.ndarray, torch.Tensor], top_k: int = 5) -> List[Tuple[int, float]]:
        # returns list of (passage_id, score)
        # Convert torch tensor to numpy if needed
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.detach().cpu().numpy()
            
        v = query_vec.reshape(1, -1)
        if self.normalize:
            v = v / (np.linalg.norm(v) + 1e-12)
        D, I = self.index.search(v.astype("float32"), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            pid = self.id_map[idx]
            results.append((pid, float(score)))
        return results
