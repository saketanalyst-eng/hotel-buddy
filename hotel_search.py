#!/usr/bin/env python3
# hotel_search.py - Simple Hotel Search Engine

import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("Loading hotel search engine...")

# Load data
with open("data/embedding_info.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
metadata = data["metadata"]
embeddings = np.load("data/embeddings.npy")
embedding_type = data["embedding_type"]

# Load appropriate encoder
if embedding_type == "sentence_bert":
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-V2')
    def encode(text):
        return encoder.encode([text])
else:
    with open("data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    def encode(text):
        return vectorizer.transform([text]).toarray()

print(f"✅ Ready! Loaded {len(chunks)} chunks using {embedding_type}")
print(f"🔍 Type 'quit' to exit\n")

def search(query, k=5):
    query_emb = encode(query)
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            results.append({
                "hotel": metadata[idx]["hotel"],
                "relevance": float(similarities[idx]),
                "info": chunks[idx][:200].replace('\n', ' ')
            })
    return results

while True:
    query = input("🔍 Your question: ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    if not query.strip():
        continue
    
    results = search(query, k=3)
    
    if not results:
        print("❌ No matching information found\n")
        continue
    
    print(f"\n📋 Top results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. 🏨 {r['hotel']} (Relevance: {r['relevance']:.2f})")
        print(f"   📄 {r['info']}...\n")
