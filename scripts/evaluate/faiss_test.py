import faiss
import numpy as np

# Create FAISS index for 768-dimensional embeddings
index = faiss.IndexFlatL2(768)

# Create 9 random embeddings (for testing)
embeddings = np.random.random((9, 768)).astype('float32')

# Add embeddings to FAISS
index.add(embeddings)

print("Total embeddings:", index.ntotal)

# Search using first embedding
query = embeddings[0].reshape(1, 768)

distances, indices = index.search(query, 3)

print("Nearest Images:")
print(indices)