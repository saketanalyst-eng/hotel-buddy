# quick_test.py - Quick verification
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Quick test for hotel search...\n")

# Load
with open("data/embedding_info.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
embeddings = np.load("data/embeddings.npy")
embedding_type = data["embedding_type"]

# Load encoder
if embedding_type == "sentence_bert":
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-V2')
else:
    with open("data/vectorizer.pkl", "rb") as f:
        encoder = pickle.load(f)
    def encode(text):
        return encoder.transform([text]).toarray()

def search(q):
    if embedding_type == "sentence_bert":
        q_emb = encoder.encode([q])
    else:
        q_emb = encoder.transform([q]).toarray()
    
    sim = cosine_similarity(q_emb, embeddings)[0]
    top = sim.argsort()[-3:][::-1]
    return [(chunks[i], sim[i]) for i in top]

# Test
test_q = "free WiFi"
print(f"Query: '{test_q}'")
results = search(test_q)
for chunk, score in results:
    print(f"  Score {score:.3f}: {chunk[:80]}...")

print("\n✅ Search working!")
