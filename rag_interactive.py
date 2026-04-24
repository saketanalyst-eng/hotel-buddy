#!/usr/bin/env python3
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
            response = f"Based on the database:\n"
            for ctx in contexts:
                response += f"\n• {ctx['hotel']}: {ctx['text'][:150]}..."
            return response
        
        # Prepare context
        context_text = "\n\n".join([
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
        print(f"\n🔍 Question: {query}")
        print("-" * 40)
        
        contexts = self.retrieve(query, k)
        
        print(f"\n📚 Retrieved {len(contexts)} relevant items:")
        for ctx in contexts:
            preview = ctx["text"][:80].replace('\n', ' ')
            print(f"   • {ctx['hotel']} (score: {ctx['relevance']:.2f})")
            print(f"     {preview}...")
        
        answer = self.generate(query, contexts)
        
        print(f"\n💡 Answer:")
        print(f"   {answer}")
        
        return answer
    
    def interactive(self):
        """Interactive Q&A session"""
        print("\n" + "=" * 50)
        print("🏨 HOTEL RAG ASSISTANT")
        print("=" * 50)
        print("Ask about hotels (type 'quit' to exit)\n")
        
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
