import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Simple FAISS-like in-memory embedding store (using PyTorch)
class EmbeddingStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, doc, embedding):
        self.documents.append(doc)
        self.embeddings.append(embedding)

    def retrieve(self, query_embedding, k=1):
        all_embeddings = torch.stack(self.embeddings)
        sims = F.cosine_similarity(query_embedding.unsqueeze(0), all_embeddings)
        topk = torch.topk(sims, k=k)
        return [self.documents[i] for i in topk.indices], topk.values

# Dummy encoder and generator
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.normalize(self.linear(x), p=2, dim=-1)

class Generator(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, vocab_size)

    def forward(self, context_embedding):
        return self.linear(context_embedding)

# Combine everything
class RAGModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, vocab_size=1000):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, hidden_dim)
        self.generator = Generator(hidden_dim, vocab_size)
        self.store = EmbeddingStore()

    def add_to_store(self, documents, embeddings):
        for doc, emb in zip(documents, embeddings):
            self.store.add(doc, emb)

    def forward(self, query, k=1):
        query_embedding = self.encoder(query)
        retrieved_docs, scores = self.store.retrieve(query_embedding, k)
        avg_embedding = torch.mean(torch.stack([self.encoder(torch.tensor(doc)) for doc in retrieved_docs]), dim=0)
        logits = self.generator(avg_embedding)
        return logits, retrieved_docs, scores

# Simulate RAG
rag = RAGModel()

# Add documents to store
docs = [torch.rand(300) for _ in range(5)]
texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]
for doc_text, doc_vector in zip(texts, docs):
    rag.add_to_store([doc_text], [doc_vector])

# Query and generate
query_vector = torch.rand(300)
logits, retrieved, sim_scores = rag(query_vector, k=2)

print("Retrieved Docs:", retrieved)
print("Logits shape (for generation):", logits.shape)
