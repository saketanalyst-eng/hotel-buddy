# step4_rag_complete.py
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

print("=" * 60)
print("🎯 STEP 4: RAG System with Ollama (Using TF-IDF Embeddings)")
print("=" * 60)

# Step 1: Load your working embeddings
print("\n📂 Loading embeddings and chunks...")

# Load chunks and metadata
with open("data/embedding_info.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
metadata = data["metadata"]

# Load embeddings
embeddings = np.load("data/embeddings.npy")

# Load vectorizer
with open("data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print(f"   ✅ Loaded {len(chunks)} chunks")
print(f"   📊 Embeddings shape: {embeddings.shape}")
print(f"   📝 Embedding type: {data.get('embedding_type', 'tfidf')}")

# Step 2: Create retrieval function
def retrieve_context(query, k=3):
    """Retrieve top-k relevant chunks"""
    # Convert query to embedding
    query_embedding = vectorizer.transform([query]).toarray()
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Prepare contexts
    contexts = []
    for idx in top_indices:
        if similarities[idx] > 0.05:  # Only include relevant results
            contexts.append({
                "text": chunks[idx],
                "hotel": metadata[idx]["hotel"],
                "similarity": float(similarities[idx]),
                "doc_id": metadata[idx]["doc_id"],
                "preview": chunks[idx][:200].replace('\n', ' ')
            })
    
    return contexts

# Step 3: Check Ollama status
def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

# Step 4: Generate answer with Ollama
def generate_answer(query, contexts, model_name="llama2"):
    """Generate answer using Ollama with grounded context"""
    
    if not contexts:
        return "No relevant information found in the hotel database."
    
    # Prepare context text
    context_text = "\n\n---\n\n".join([
        f"Hotel: {ctx['hotel']}\nInformation: {ctx['text']}" 
        for ctx in contexts
    ])
    
    # Create prompt with grounding instructions
    prompt = f"""You are a hotel customer service assistant. Answer the question using ONLY the information provided below.

INFORMATION FROM HOTEL DATABASE:
{context_text}

QUESTION: {query}

RULES:
1. ONLY use information from the database above
2. Always mention which hotel the information comes from
3. If the information is not in the database, say "I don't have that information"
4. Be concise and helpful
5. Do not add any external knowledge

ANSWER:"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # Lower = more factual
                "top_k": 40,
                "top_p": 0.9
            },
            timeout=30
        )
        
        if response.status_code == 200:
            answer = response.json()["response"]
            return answer.strip()
        else:
            return f"Ollama error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return "❌ Ollama not running. Please start Ollama first."
    except Exception as e:
        return f"Error: {str(e)}"

# Step 5: Complete RAG pipeline
def rag_query(query, k=3, model_name="llama2"):
    """Complete RAG pipeline"""
    print(f"\n🔍 Processing: {query}")
    print("-" * 50)
    
    # Retrieve
    contexts = retrieve_context(query, k)
    
    print(f"\n📚 Retrieved {len(contexts)} relevant chunks:")
    for i, ctx in enumerate(contexts, 1):
        print(f"   {i}. {ctx['hotel']} (Relevance: {ctx['similarity']:.3f})")
        print(f"      {ctx['preview'][:100]}...")
    
    # Generate
    if check_ollama():
        answer = generate_answer(query, contexts, model_name)
    else:
        answer = "⚠️ Ollama not available. Using fallback mode.\n\nBased on retrieved information:\n"
        for ctx in contexts:
            answer += f"\n• {ctx['hotel']}: {ctx['preview'][:150]}..."
    
    print(f"\n💡 ANSWER:")
    print(f"   {answer}")
    
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "retrieved_hotels": list(set([c["hotel"] for c in contexts]))
    }

# Step 6: Test the RAG system
print("\n" + "=" * 60)
print("🧪 Testing RAG Pipeline")
print("=" * 60)

# First, check Ollama
ollama_available = check_ollama()
if ollama_available:
    print("\n✅ Ollama is running!")
    
    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        if models:
            print(f"📦 Available models: {[m['name'] for m in models]}")
            default_model = models[0]['name'].split(':')[0]
        else:
            default_model = "llama2"
            print(f"⚠️ No models found. Run: ollama pull llama2")
    except:
        default_model = "llama2"
else:
    print("\n⚠️ Ollama not detected!")
    print("   To install Ollama:")
    print("   1. Visit https://ollama.ai")
    print("   2. Download and install")
    print("   3. Run: ollama serve")
    print("   4. Pull model: ollama pull llama2")
    print("\n   For now, running in DEMO mode...")
    default_model = "demo"

# Test queries
test_queries = [
    "Which hotels have free WiFi?",
    "What is the cancellation policy?",
    "Do any hotels allow pets?",
    "Suggest hotels near the beach",
    "What time is check-in and check-out?"
]

print("\n" + "=" * 60)
print("📋 RAG SYSTEM TEST RESULTS")
print("=" * 60)

results_list = []
for query in test_queries:
    result = rag_query(query, k=3, model_name=default_model if ollama_available else "demo")
    results_list.append(result)
    print("\n" + "=" * 60)

# Step 7: Save results
print("\n💾 Saving results...")
with open("outputs/rag_results.json", "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2, ensure_ascii=False)
print("   ✅ Saved: outputs/rag_results.json")

# Step 8: Create interactive RAG system
print("\n💾 Creating interactive RAG system...")

rag_interactive = '''#!/usr/bin/env python3
"""
Hotel RAG System - Interactive Q&A
Uses: TF-IDF Retrieval + Ollama Generation
"""

import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
import sys

class HotelRAG:
    def __init__(self):
        print("🚀 Loading Hotel RAG System...")
        
        # Load data
        with open("data/embedding_info.json", "r") as f:
            self.data = json.load(f)
        
        self.chunks = self.data["chunks"]
        self.metadata = self.data["metadata"]
        self.embeddings = np.load("data/embeddings.npy")
        
        with open("data/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} hotel information chunks")
        
        # Check Ollama
        self.ollama_available = self._check_ollama()
        if self.ollama_available:
            self.model = self._get_default_model()
            print(f"✅ Ollama ready (model: {self.model})")
        else:
            print("⚠️ Ollama not available. Using fallback mode.")
    
    def _check_ollama(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _get_default_model(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = response.json().get("models", [])
            return models[0]['name'].split(':')[0] if models else "llama2"
        except:
            return "llama2"
    
    def retrieve(self, query, k=3):
        """Retrieve relevant hotel information"""
        q_emb = self.vectorizer.transform([query]).toarray()
        similarities = cosine_similarity(q_emb, self.embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.05:
                contexts.append({
                    "text": self.chunks[idx],
                    "hotel": self.metadata[idx]["hotel"],
                    "relevance": float(similarities[idx])
                })
        return contexts
    
    def generate(self, query, contexts):
        """Generate answer using Ollama"""
        if not contexts:
            return "No relevant hotel information found."
        
        if not self.ollama_available:
            # Fallback response
            response = f"Based on the database:\\n"
            for ctx in contexts:
                response += f"\\n• {ctx['hotel']}: {ctx['text'][:150]}..."
            return response
        
        # Prepare context
        context_text = "\\n\\n".join([
            f"[{ctx['hotel']}] {ctx['text']}" for ctx in contexts
        ])
        
        prompt = f"""Answer using ONLY the hotel information below.

HOTEL INFORMATION:
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
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask(self, query, k=3):
        """Ask a question"""
        print(f"\\n🔍 Question: {query}")
        print("-" * 40)
        
        contexts = self.retrieve(query, k)
        
        print(f"\\n📚 Retrieved {len(contexts)} relevant items:")
        for ctx in contexts:
            preview = ctx["text"][:80].replace('\\n', ' ')
            print(f"   • {ctx['hotel']} (score: {ctx['relevance']:.2f})")
            print(f"     {preview}...")
        
        answer = self.generate(query, contexts)
        
        print(f"\\n💡 Answer:")
        print(f"   {answer}")
        
        return answer
    
    def interactive(self):
        """Interactive Q&A session"""
        print("\\n" + "=" * 50)
        print("🏨 HOTEL RAG ASSISTANT")
        print("=" * 50)
        print("Ask about hotels (type 'quit' to exit)\\n")
        
        while True:
            query = input("❓ You: ")
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue
            
            self.ask(query)
            print()

if __name__ == "__main__":
    rag = HotelRAG()
    rag.interactive()
'''

with open("rag_interactive.py", "w", encoding="utf-8") as f:
    f.write(rag_interactive)
print("   ✅ Created: rag_interactive.py")

# Step 9: Create evaluation script
print("\n💾 Creating evaluation script...")

eval_script = '''# evaluate_rag.py - Evaluate RAG performance
import json
from rag_interactive import HotelRAG

def calculate_precision_at_k(query, expected_hotels, k=3):
    """Calculate Precision@k metric"""
    rag = HotelRAG()
    contexts = rag.retrieve(query, k)
    
    retrieved_hotels = [ctx["hotel"] for ctx in contexts]
    relevant_count = sum(1 for h in retrieved_hotels if h in expected_hotels)
    
    precision = relevant_count / k if k > 0 else 0
    
    return {
        "query": query,
        "precision_at_k": precision,
        "retrieved": retrieved_hotels,
        "expected": expected_hotels
    }

# Test cases
test_cases = [
    {
        "query": "Which hotels have free WiFi?",
        "expected_hotels": ["Seaside Paradise Resort", "City Central Hotel", "Family Fun Resort"]
    },
    {
        "query": "Hotels near beach",
        "expected_hotels": ["Seaside Paradise Resort", "Ocean View Villas"]
    },
    {
        "query": "Pet friendly hotels",
        "expected_hotels": ["City Central Hotel", "Sunny Garden Inn", "Ocean View Villas"]
    }
]

print("=" * 50)
print("RAG SYSTEM EVALUATION")
print("=" * 50)

results = []
for test in test_cases:
    result = calculate_precision_at_k(test["query"], test["expected_hotels"], k=3)
    results.append(result)
    
    print(f"\\nQuery: {result['query']}")
    print(f"Precision@3: {result['precision_at_k']:.2f}")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Expected: {result['expected']}")

avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
print(f"\\n📊 Average Precision@3: {avg_precision:.2f}")

# Save results
with open("outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\n✅ Results saved to outputs/evaluation_results.json")
'''

with open("evaluate_rag.py", "w", encoding="utf-8") as f:
    f.write(eval_script)
print("   ✅ Created: evaluate_rag.py")

print("\n" + "=" * 60)
print("✅ STEP 4 COMPLETE!")
print("=" * 60)

print("\n📊 Summary:")
print(f"   🔍 Retrieval: TF-IDF (Working without PyTorch)")
print(f"   🤖 LLM: Ollama {'Available' if ollama_available else 'Not Available'}")
print(f"\n📁 Files Created:")
print(f"   • rag_interactive.py - Interactive Q&A system")
print(f"   • evaluate_rag.py - Evaluation script")
print(f"   • outputs/rag_results.json - Test results")

print("\n🚀 Next Steps:")

if ollama_available:
    print("   ✅ Ollama is running!")
    print("   1. Run interactive RAG: python rag_interactive.py")
    print("   2. Evaluate system: python evaluate_rag.py")
else:
    print("   ⚠️ Install Ollama for better answers:")
    print("   1. Download from https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull model: ollama pull llama2")
    print("   4. Run: python rag_interactive.py")

print("\n💡 Hallucination Control:")
print("   • Prompts force context-only answers")
print("   • Temperature set to 0.3 (factual)")
print("   • Explicit rules to use ONLY retrieved data")
print("   • Fallback when no information found")

# Step 5: Hallucination Control Summary
print("\n" + "=" * 60)
print("🎯 HALLUCINATION CONTROL IMPLEMENTED")
print("=" * 60)
print("""
1. GROUNDED PROMPTS:
   - LLM instructed to use ONLY retrieved context
   - Explicit rules: "Do not add external knowledge"

2. TEMPERATURE CONTROL:
   - Temperature = 0.3 (reduces creativity/randomness)
   - More factual, less hallucination

3. CONTEXT VERIFICATION:
   - If no relevant chunks found → "No information found"
   - Empty context handling prevents hallucination

4. SOURCE ATTRIBUTION:
   - Answer must mention which hotel provided info
   - Traceable back to source documents

5. CONFIDENCE SCORING:
   - Similarity scores shown for transparency
   - Low relevance results filtered out
""")

print("\n✅ RAG System Ready for Demo!")