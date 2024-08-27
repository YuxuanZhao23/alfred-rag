from alfred import Client
import os
import torch
from copy import deepcopy as dc
from wsindex.embeddings.embeddings_manager import EmbeddingManager

DEFAULT_TEMPLATE = "Based on the following {} documents, {}\n"

class Embeddings:
    def __init__(self,
                 dataset,
                 name,
                 query_model="text-davinci-003",
                 query_model_type="openai",
                 emb_model="text-embedding-ada-002",
                 emb_model_type="openai",
                 save_path="./",
                 ):

        self.emb_client = Client(model_type=emb_model_type, model=emb_model)
        self.dataset = dataset
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        emb_path = os.path.join(save_path, f"{name}.embs")
        if not os.path.exists(emb_path):
            print("Embeddings not found, generating embeddings...")
            self.embeddings = self.emb_client.encode(dataset)
            torch.save(self.embeddings, emb_path)
        else:
            print("Embeddings found, loading embeddings...")
            self.embeddings = torch.load(emb_path)
        self.embedding_manager = EmbeddingManager(self.embeddings)

        self.query_client  = Client(model_type=query_model_type, model=query_model)

    def query(self, query, k=10, template=DEFAULT_TEMPLATE):
        query_embeddings = self.emb_client.encode([query])
        distances, indices = self.embedding_manager.search(query_embeddings, k)
        prompt = dc(template).format(len(indices), query) + '\n'.join([f"{idx}: " + self.dataset[i] for idx, i in enumerate(indices)])
        print(prompt)
        return self.query_client(prompt, temperature=0.7, top_p=0.9, max_tokens=100)



