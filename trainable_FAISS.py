import torch
import torch.nn as nn
import faiss
import numpy as np

class TrainableFAISS(nn.Module):
    def __init__(self, embed_dim, num_docs, index_type='Flat'):
        super(TrainableFAISS, self).__init__()
        self.embed_dim = embed_dim
        self.num_docs = num_docs

        # Initialize document embeddings randomly
        self.doc_embeddings = nn.Embedding(num_docs, embed_dim)
        self.index = faiss.IndexFlatL2(embed_dim)  # Use FAISS for similarity search

    def forward(self, query, top_k=3):
        # Query embedding (trainable)
        query_embedding = query.unsqueeze(0)  # Add batch dimension
        distances, indices = self.search(query_embedding, top_k)

        return distances, indices

    def search(self, query_embedding, top_k=3):
        # Convert query embedding to numpy for FAISS
        query_embedding_np = query_embedding.detach().cpu().numpy().astype('float32')

        # Search for the nearest neighbors
        distances, indices = self.index.search(query_embedding_np, top_k)

        return distances, indices

    def update_index(self):
        # Update FAISS index with the current document embeddings
        doc_embeddings_np = self.doc_embeddings.weight.detach().cpu().numpy().astype('float32')
        self.index.add(doc_embeddings_np)

    def train_embeddings(self, query, target_doc_idx, learning_rate=1e-3):
        # Train embeddings using a simple retrieval-based loss (contrastive)
        query_embedding = query.unsqueeze(0)
        self.update_index()

        # Get the closest document based on FAISS search
        distances, indices = self.search(query_embedding)
        closest_doc_idx = indices[0][0]

        # Contrastive loss (negative log-likelihood)
        loss = torch.nn.functional.cross_entropy(self.doc_embeddings(closest_doc_idx), target_doc_idx)
        
        # Backpropagate and update document embeddings
        self.zero_grad()
        loss.backward()
        self.doc_embeddings.weight.data -= learning_rate * self.doc_embeddings.weight.grad.data

        return loss

# Example usage
num_docs = 1000
embed_dim = 128
model = TrainableFAISS(embed_dim=embed_dim, num_docs=num_docs)

# Example query (batch size 1)
query = torch.randn(embed_dim)

# Add documents to the FAISS index
model.update_index()

# Search for top 3 closest documents
distances, indices = model(query, top_k=3)
print("Top 3 nearest docs:", indices)

# Train embeddings for a document
target_doc_idx = torch.tensor([5])
loss = model.train_embeddings(query, target_doc_idx)
print("Training loss:", loss.item())
