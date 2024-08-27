import torch
import pickle
from wsindex.embeddings.embedding_index import EmbeddingIndex
import numpy as np

class EmbeddingManager:
    def __init__(self, embeddings):
        self.index = EmbeddingIndex(torch.stack(embeddings, dim=0).numpy())
        self.ids = torch.arange(len(embeddings))
        self.embeddings = embeddings

    def add(self, embeddings: torch.Tensor):
        self.index.add(embeddings)
        self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.ids = torch.cat([self.ids, torch.arange(self.ids.size(0), self.ids.size(0) + embeddings.size(0))], dim=0)

    def remove(self, ids):
        self.index.remove(ids)
        self.embeddings = self.embeddings[~ids]
        self.ids = self.ids[~ids]

    def search(self, query_embeddings, k=10):
        query_embeddings = torch.stack(query_embeddings, dim=0).numpy()
        return self.index.search(query_embeddings, k)

    def save(self, path):
        # use pickle to save the EmbeddingManager object
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)