# rag_with_vectordb_fixed.py
import json
import numpy as np
import pickle
import requests
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏨 RAG SYSTEM WITH VECTOR DATABASE (Compatible Version)")
print("=" * 70)

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================
# PART 1: LOAD DATA
# ============================================
print("\n📂 Loading data...")

# Load chunks and metadata
with open("data/embedding_info.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
metadata = data["metadata"]
print(f"   ✅ Loaded {len(chunks)} chunks")

# Load embeddings
embeddings = np.load("data/embeddings.npy").astype('float32')
print(f"   ✅ Loaded embeddings shape: {embeddings.shape}")

# Load vectorizer for queries
with open("data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ============================================
# PART 2: TRY FAISS WITH ERROR HANDLING
# ============================================
print("\n🔧 Setting up Vector Database...")

USE_FAISS = False
try:
    import faiss
    print("   ✅ FAISS package found")
    
    # Check NumPy version compatibility
    if np.__version__ >= '1.24.0':
        print(f"   ⚠️ NumPy {np.__version__} may have issues with FAISS")
        print("   📌 Using sklearn fallback for stability")
        USE_FAISS = False
    else:
        USE_FAISS = True
        print(f"   ✅ NumPy {np.__version__} compatible")
        
except ImportError as e:
    print(f"   ⚠️ FAISS not available: {e}")
    print("   📌 Using sklearn for vector search")
    USE_FAISS = False

# ============================================
# PART 3: VECTOR SEARCH (Dual Implementation)
# ============================================
def vector_search_sklearn(query, k=5):
    """Search using sklearn (always works)"""
    query_embedding = vectorizer.transform([query]).toarray()
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    
    contexts = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            contexts.append({
                "text": chunks[idx],
                "hotel": metadata[idx]["hotel"],
                "similarity_score": float(similarities[idx]),
                "doc_id": metadata[idx]["doc_id"],
                "chunk_id": metadata[idx]["chunk_id"]
            })
    return contexts

def vector_search_faiss(query, k=5):
    """Search using FAISS if available"""
    if not USE_FAISS:
        return vector_search_sklearn(query, k)
    
    try:
        query_embedding = vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = index.search(query_embedding, k)
        
        contexts = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(chunks):
                similarity = 1 / (1 + distances[0][i])
                contexts.append({
                    "text": chunks[idx],
                    "hotel": metadata[idx]["hotel"],
                    "similarity_score": float(similarity),
                    "distance": float(distances[0][i]),
                    "doc_id": metadata[idx]["doc_id"],
                    "chunk_id": metadata[idx]["chunk_id"]
                })
        return contexts
    except Exception as e:
        print(f"   FAISS error: {e}, falling back to sklearn")
        return vector_search_sklearn(query, k)

# Choose search method
if USE_FAISS:
    try:
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"   ✅ FAISS index created with {index.ntotal} vectors")
        vector_search = vector_search_faiss
        search_method = "FAISS"
    except Exception as e:
        print(f"   ⚠️ FAISS initialization failed: {e}")
        vector_search = vector_search_sklearn
        search_method = "sklearn"
else:
    vector_search = vector_search_sklearn
    search_method = "sklearn"

print(f"   🔍 Using {search_method} for vector search")

# ============================================
# PART 4: CREATE SIMPLE VECTOR DB CLASS
# ============================================
class SimpleVectorDB:
    """Simple vector database wrapper"""
    
    def __init__(self, chunks, embeddings, metadata, vectorizer):
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = metadata
        self.vectorizer = vectorizer
        self.search_method = search_method
        
        if USE_FAISS and 'index' in globals():
            self.index = index
            self.use_faiss = True
        else:
            self.use_faiss = False
        
        print(f"   ✅ VectorDB ready ({len(chunks)} vectors)")
    
    def search(self, query, k=5):
        """Search similar vectors"""
        if self.use_faiss:
            return vector_search_faiss(query, k)
        else:
            return vector_search_sklearn(query, k)
    
    def get_stats(self):
        """Get database statistics"""
        return {
            "total_vectors": len(self.chunks),
            "dimension": self.embeddings.shape[1],
            "search_method": self.search_method,
            "metadata_fields": list(self.metadata[0].keys()) if self.metadata else []
        }

# Initialize vector DB
vectordb = SimpleVectorDB(chunks, embeddings, metadata, vectorizer)

# ============================================
# PART 5: OLLAMA INTEGRATION
# ============================================
def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False

def generate_answer(query, contexts, model_name="llama2"):
    """Generate answer using Ollama"""
    if not contexts:
        return "No relevant hotel information found in database.", 0.0
    
    # Prepare context
    context_text = "\n\n".join([
        f"[{ctx['hotel']}] {ctx['text']}" for ctx in contexts[:3]
    ])
    
    prompt = f"""Answer using ONLY the hotel information below.

INFORMATION:
{context_text}

QUESTION: {query}

RULES:
- Only use the information above
- Mention hotel names
- If not found, say "Not available"

ANSWER:"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            avg_score = sum(c["similarity_score"] for c in contexts[:3]) / 3
            return answer, avg_score
        else:
            return fallback_answer(query, contexts), 0.5
    except:
        return fallback_answer(query, contexts), 0.5

def fallback_answer(query, contexts):
    """Fallback when Ollama unavailable"""
    answer = "Based on retrieved information:\n"
    for ctx in contexts[:2]:
        preview = ctx["text"][:150].replace("\n", " ")
        answer += f"\n• {ctx['hotel']}: {preview}..."
    return answer

# ============================================
# PART 6: COMPLETE RAG PIPELINE
# ============================================
def rag_query(query, k=5):
    """Complete RAG pipeline"""
    start_time = time.time()
    
    # Search vector DB
    contexts = vectordb.search(query, k)
    
    # Generate answer
    ollama_available = check_ollama()
    if ollama_available:
        answer, confidence = generate_answer(query, contexts)
    else:
        answer = fallback_answer(query, contexts)
        confidence = 0.5
    
    retrieval_time = time.time() - start_time
    
    return {
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "contexts_used": len(contexts),
        "retrieval_time": round(retrieval_time, 3),
        "sources": [{"hotel": c["hotel"], "score": c["similarity_score"]} for c in contexts[:3]],
        "search_method": vectordb.search_method
    }

# ============================================
# PART 7: DEMONSTRATION
# ============================================
def run_demo():
    """Run demonstration"""
    print("\n" + "=" * 70)
    print("📋 VECTOR DATABASE STATISTICS")
    print("=" * 70)
    
    stats = vectordb.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("🚀 RAG SYSTEM DEMO")
    print("=" * 70)
    
    test_queries = [
        "Which hotels have free WiFi?",
        "What is the cancellation policy?",
        "Hotels near the beach",
        "Pet friendly hotels"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 50)
        
        result = rag_query(query, k=5)
        
        print(f"\n📚 Retrieved from {result['search_method']}:")
        for i, src in enumerate(result['sources'], 1):
            print(f"   {i}. {src['hotel']} (Score: {src['score']:.3f})")
        
        print(f"\n💡 Answer:")
        print(f"   {result['answer']}")
        
        print(f"\n⏱️ Time: {result['retrieval_time']}s | Confidence: {result['confidence']:.2f}")
        print("=" * 50)

# ============================================
# PART 8: INTERACTIVE MODE
# ============================================
def interactive_mode():
    """Interactive Q&A"""
    print("\n" + "=" * 70)
    print("🏨 HOTEL RAG SYSTEM - INTERACTIVE MODE")
    print("=" * 70)
    print(f"🔍 Vector DB: {vectordb.search_method} search")
    print(f"📊 Total vectors: {vectordb.get_stats()['total_vectors']}")
    print("💡 Type 'quit' to exit\n")
    
    while True:
        query = input("❓ You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        result = rag_query(query, k=5)
        
        print(f"\n🤖 Assistant: {result['answer']}")
        print(f"\n[Sources: {', '.join([s['hotel'] for s in result['sources']])}]")
        print(f"[Time: {result['retrieval_time']}s | Confidence: {result['confidence']:.2f}]\n")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Print system info
    print(f"\n📊 System Info:")
    print(f"   NumPy version: {np.__version__}")
    print(f"   Search method: {search_method}")
    
    # Run demo
    run_demo()
    
    # Interactive mode
    print("\n" + "=" * 70)
    choice = input("🎮 Start interactive mode? (y/n): ").strip().lower()
    
    if choice == 'y':
        interactive_mode()
    else:
        print("\n✅ RAG with Vector DB ready!")
        print("   Run: python rag_with_vectordb_fixed.py")