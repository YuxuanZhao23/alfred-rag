import torch
import faiss
import torch
import numpy as np

class EmbeddingIndex:
    def __init__(self, embeddings: np.ndarray,
                 index_type="IVF",
                 use_gpu=True):
        self.embedding_dim, self.nlist = embeddings.shape[-1], embeddings.shape[0]
        self.index_type = index_type
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.build_index(embeddings)

    def build_index(self, embeddings):

        if self.index_type == "IVF":
            index = faiss.IndexFlatIP(self.embedding_dim)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "IVF_SQ8":
            index = faiss.IndexFlatL2(self.embedding_dim)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFSQ8(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported index type: {}".format(self.index_type))
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(index)

        self.index.train(embeddings)
        self.index.add(embeddings)

    def add(self, embeddings: torch.Tensor):
        if self.index is None:
            self.build_index(embeddings)

        if not self.use_gpu:
            embeddings = embeddings.cpu().numpy()
        else:
            embeddings = np.array(embeddings)

        self.index.add(embeddings)

    def remove(self, ids):
        if self.index is None:
            raise ValueError("Index has not been built yet")

        if self.use_gpu:
            ids = ids.cuda()
        else:
            ids = ids.cpu().numpy()

        self.index.remove_ids(ids)

    def search(self, query_embeddings, k=10):
        if self.index is None:
            raise ValueError("Index has not been built yet")

        if self.use_gpu:
            query_embeddings = np.array(query_embeddings)
        else:
            query_embeddings = np.array(query_embeddings)

        distances, indices = self.index.search(query_embeddings, k)

        # Force the first result to be the only result
        distances = distances[0]
        indices = indices[0]

        mask = indices != -1
        distances = distances[mask]
        indices = indices[mask]
        return distances, indices


    def save(self, filepath):
        if self.index is None:
            raise ValueError("Index has not been built yet")

        faiss.write_index(self.index, filepath)

    def load(self, filepath):
        self.index = faiss.read_index(filepath)

        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
