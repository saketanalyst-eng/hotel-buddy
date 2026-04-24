# evaluate_rag.py - Evaluate RAG performance
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
    
    print(f"\nQuery: {result['query']}")
    print(f"Precision@3: {result['precision_at_k']:.2f}")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Expected: {result['expected']}")

avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
print(f"\n📊 Average Precision@3: {avg_precision:.2f}")

# Save results
with open("outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ Results saved to outputs/evaluation_results.json")
