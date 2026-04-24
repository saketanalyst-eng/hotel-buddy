# step3_complete_windows_fix.py
import json
import numpy as np
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🎯 STEP 3: Embeddings (Windows Compatible)")
print("=" * 60)

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("src", exist_ok=True)

# Step 1: Load chunks
print("\n📂 Loading chunks...")
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

chunks = chunk_data["chunks"]
metadata = chunk_data["metadata"]
print(f"   ✅ Loaded {len(chunks)} chunks")

# Step 2: Choose embedding method based on available packages
print("\n🤖 Setting up embeddings...")

# Try different embedding methods in order
USE_SBERT = False
USE_TFIDF = True  # Fallback to TF-IDF

try:
    # Try Sentence-BERT first
    from sentence_transformers import SentenceTransformer
    print("   Trying Sentence-BERT...")
    model = SentenceTransformer('all-MiniLM-L6-V2')
    USE_SBERT = True
    USE_TFIDF = False
    print("   ✅ Using Sentence-BERT (Deep Learning)")
except Exception as e:
    print(f"   ⚠️ Sentence-BERT failed: {str(e)[:50]}...")
    print("   📌 Using TF-IDF (Keyword-based, no PyTorch needed)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_TFIDF = True

# Step 3: Generate embeddings
print("\n🔢 Generating embeddings...")
start_time = time.time()

if USE_SBERT:
    # Deep learning embeddings
    print("   Method: Sentence-BERT (Semantic understanding)")
    batch_size = 16
    embeddings_list = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings_list.extend(batch_embeddings)
        print(f"   Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    embeddings = np.array(embeddings_list).astype('float32')
    embedding_type = "sentence_bert"
    
else:
    # TF-IDF embeddings (No neural networks)
    print("   Method: TF-IDF (Keyword-based, 100% compatible)")
    vectorizer = TfidfVectorizer(
        max_features=384,
        stop_words='english',
        ngram_range=(1, 2)
    )
    embeddings = vectorizer.fit_transform(chunks).toarray().astype('float32')
    embedding_type = "tfidf"
    
    # Save vectorizer for later
    import pickle
    with open("data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

end_time = time.time()
print(f"\n   ✅ Generated {len(embeddings)} embeddings")
print(f"   ⏱️  Time: {end_time - start_time:.2f} seconds")
print(f"   📊 Shape: {embeddings.shape}")
print(f"   📝 Type: {embedding_type}")

# Step 4: Save embeddings
print("\n💾 Saving embeddings...")

np.save("data/embeddings.npy", embeddings)
print(f"   ✅ Saved: data/embeddings.npy")

# Save metadata
embedding_info = {
    "embedding_type": embedding_type,
    "num_chunks": len(chunks),
    "embedding_dimension": embeddings.shape[1],
    "chunks": chunks,
    "metadata": metadata,
    "creation_time": time.time()
}

with open("data/embedding_info.json", "w", encoding="utf-8") as f:
    json.dump(embedding_info, f, indent=2)
print(f"   ✅ Saved: data/embedding_info.json")

# Step 5: Create search function
print("\n🔍 Creating search function...")

if USE_SBERT:
    def get_embedding(text):
        return model.encode([text])
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    with open("data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    def get_embedding(text):
        return vectorizer.transform([text]).toarray()

def search_similar(query, k=5):
    """Search for similar chunks"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_emb = get_embedding(query)
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            results.append({
                "chunk": chunks[idx],
                "similarity": float(similarities[idx]),
                "hotel": metadata[idx]["hotel"],
                "preview": chunks[idx][:150].replace('\n', ' ')
            })
    return results

# Step 6: Test search
print("\n🧪 Testing search...")

test_queries = [
    "free WiFi",
    "cancellation policy",
    "beach hotel",
    "pet friendly",
    "breakfast included"
]

print("\n" + "=" * 60)
for query in test_queries:
    print(f"\n❓ Query: '{query}'")
    print("-" * 40)
    
    results = search_similar(query, k=3)
    
    if not results:
        print("   No results found")
    else:
        for i, res in enumerate(results, 1):
            print(f"\n{i}. 🏨 {res['hotel']} (score: {res['similarity']:.3f})")
            print(f"   📄 {res['preview'][:100]}...")
    print("-" * 40)

# Step 7: Create search engine script
print("\n💾 Creating search engine...")

search_engine = '''#!/usr/bin/env python3
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
print(f"🔍 Type 'quit' to exit\\n")

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
                "info": chunks[idx][:200].replace('\\n', ' ')
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
        print("❌ No matching information found\\n")
        continue
    
    print(f"\\n📋 Top results:\\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. 🏨 {r['hotel']} (Relevance: {r['relevance']:.2f})")
        print(f"   📄 {r['info']}...\\n")
'''

with open("hotel_search.py", "w", encoding="utf-8") as f:
    f.write(search_engine)
print("   ✅ Created: hotel_search.py")

# Step 8: Create quick test
quick_test = '''# quick_test.py - Quick verification
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Quick test for hotel search...\\n")

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

print("\\n✅ Search working!")
'''

with open("quick_test.py", "w", encoding="utf-8") as f:
    f.write(quick_test)
print("   ✅ Created: quick_test.py")

print("\n" + "=" * 60)
print("✅ STEP 3 COMPLETE!")
print("=" * 60)

print("\n📊 Summary:")
print(f"   📄 Chunks processed: {len(chunks)}")
print(f"   🔢 Embedding type: {embedding_type}")
print(f"   📐 Dimension: {embeddings.shape[1]}")
print(f"\n📁 Files created:")
print(f"   • data/embeddings.npy - Vector embeddings")
print(f"   • data/embedding_info.json - Metadata")
print(f"   • hotel_search.py - Search engine")
print(f"   • quick_test.py - Quick test")

print("\n🚀 Test the system:")
print("   python quick_test.py")
print("   python hotel_search.py")

if embedding_type == "tfidf":
    print("\n💡 Note: Using TF-IDF (keyword-based search)")
    print("   This works without PyTorch!")
    print("   For semantic search, fix PyTorch with:")
    print("   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu")