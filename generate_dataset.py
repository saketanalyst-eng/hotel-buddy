# step1_generate_dataset.py
import json
import os

# Create directories first
os.makedirs("data", exist_ok=True)
os.makedirs("src", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("📁 STEP 1: Generating Hotel Dataset")
print("=" * 60)

# Hotel data template
hotels_data = [
    {
        "name": "Seaside Paradise Resort",
        "location": "Coastal Road, Malibu Beach",
        "distance_to_beach": "50 meters",
        "rating": 4.7,
        "amenities": ["free WiFi", "breakfast buffet", "pool", "spa", "parking", "beach access"],
        "policies": {
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 48 hours before check-in",
            "pet_policy": "Pets not allowed"
        },
        "reviews": [
            "Amazing beach view from the room!",
            "Great breakfast selection",
            "Staff was very friendly"
        ],
        "description": "Luxury beachfront resort with stunning ocean views. Located just steps from Malibu Beach."
    },
    {
        "name": "City Central Hotel",
        "location": "Downtown, 5th Avenue",
        "distance_to_beach": "15 km",
        "rating": 4.2,
        "amenities": ["free WiFi", "business center", "fitness center", "restaurant", "room service", "free breakfast"],
        "policies": {
            "check_in": "2:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 24 hours before check-in",
            "pet_policy": "Small pets allowed ($50 fee)"
        },
        "reviews": [
            "Perfect for business travelers",
            "Clean rooms, good location",
            "Breakfast could be better"
        ],
        "description": "Modern hotel in the heart of downtown. Close to shopping and business district."
    },
    {
        "name": "Sunny Garden Inn",
        "location": "Suburb Heights, Park Street",
        "distance_to_beach": "8 km",
        "rating": 4.5,
        "amenities": ["free WiFi", "free breakfast", "garden", "bicycle rental", "free parking"],
        "policies": {
            "check_in": "1:00 PM",
            "check_out": "10:00 AM",
            "cancellation": "Free cancellation up to 7 days before check-in",
            "pet_policy": "Pets allowed (no extra fee)"
        },
        "reviews": [
            "Beautiful garden and peaceful atmosphere",
            "Very reasonable prices",
            "Friendly hosts"
        ],
        "description": "Cozy garden hotel with home-like atmosphere. Perfect for families and nature lovers."
    },
    {
        "name": "Luxury Grand Plaza",
        "location": "Waterfront Drive, Marina Bay",
        "distance_to_beach": "200 meters",
        "rating": 4.9,
        "amenities": ["free high-speed WiFi", "premium breakfast", "infinity pool", "luxury spa", "valet parking", "concierge", "beach shuttle"],
        "policies": {
            "check_in": "4:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "No cancellation within 14 days",
            "pet_policy": "Service animals only"
        },
        "reviews": [
            "Absolute luxury experience!",
            "Best hotel I've ever stayed at",
            "Amazing infinity pool overlooking the bay"
        ],
        "description": "5-star luxury hotel with world-class amenities. Prime waterfront location."
    },
    {
        "name": "Budget Stay Hostel",
        "location": "Backpacker Lane, Near Train Station",
        "distance_to_beach": "12 km",
        "rating": 4.0,
        "amenities": ["free WiFi", "shared kitchen", "lockers", "common lounge", "free coffee"],
        "policies": {
            "check_in": "11:00 AM",
            "check_out": "10:00 AM",
            "cancellation": "Free cancellation up to 24 hours before check-in",
            "pet_policy": "No pets allowed"
        },
        "reviews": [
            "Great value for money",
            "Clean and friendly",
            "Perfect for backpackers"
        ],
        "description": "Affordable accommodation for budget travelers. Social atmosphere and clean facilities."
    },
    {
        "name": "Ocean View Villas",
        "location": "Cliffside Road, Pacific Heights",
        "distance_to_beach": "100 meters",
        "rating": 4.8,
        "amenities": ["free WiFi", "private pool", "beach access", "parking", "fully equipped kitchen", "BBQ area"],
        "policies": {
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 30 days before check-in",
            "pet_policy": "Pets allowed with deposit"
        },
        "reviews": [
            "Breathtaking ocean views",
            "Perfect for groups",
            "Very private and quiet"
        ],
        "description": "Luxury private villas perched on the cliffs. Each villa has panoramic ocean views."
    },
    {
        "name": "Airport Transit Hotel",
        "location": "Terminal Road, Airport Area",
        "distance_to_beach": "20 km",
        "rating": 4.1,
        "amenities": ["free WiFi", "24-hour reception", "shuttle service", "basic breakfast", "soundproof rooms"],
        "policies": {
            "check_in": "Flexible (any time)",
            "check_out": "Flexible (any time)",
            "cancellation": "Free cancellation up to 6 hours before check-in",
            "pet_policy": "Pets not allowed"
        },
        "reviews": [
            "Convenient for early flights",
            "Basic but clean",
            "Shuttle service is reliable"
        ],
        "description": "Convenient airport hotel perfect for transit passengers. 24/7 service and airport shuttle."
    },
    {
        "name": "Mountain View Retreat",
        "location": "Hill Top Road, Pine Forest",
        "distance_to_beach": "25 km",
        "rating": 4.6,
        "amenities": ["free WiFi", "breakfast included", "hiking trails", "bonfire area", "mountain views"],
        "policies": {
            "check_in": "2:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 14 days before check-in",
            "pet_policy": "Pets allowed"
        },
        "reviews": [
            "Stunning mountain scenery",
            "Great hiking opportunities",
            "Cozy and warm atmosphere"
        ],
        "description": "Peaceful retreat in the mountains. Perfect for nature enthusiasts and hikers."
    },
    {
        "name": "Family Fun Resort",
        "location": "Resort Drive, Kid's Paradise",
        "distance_to_beach": "5 km",
        "rating": 4.4,
        "amenities": ["free WiFi", "free breakfast", "children's pool", "playground", "family rooms", "game room", "free parking"],
        "policies": {
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 72 hours before check-in",
            "pet_policy": "No pets"
        },
        "reviews": [
            "Kids loved the playground",
            "Very family-friendly",
            "Great activities for children"
        ],
        "description": "Resort designed especially for families. Kids stay free and plenty of activities."
    },
    {
        "name": "Business Elite Suites",
        "location": "Financial District, Tower Road",
        "distance_to_beach": "18 km",
        "rating": 4.7,
        "amenities": ["free high-speed WiFi", "business lounge", "meeting rooms", "free breakfast", "airport transfer", "concierge"],
        "policies": {
            "check_in": "12:00 PM",
            "check_out": "1:00 PM",
            "cancellation": "Free cancellation up to 72 hours before check-in",
            "pet_policy": "No pets"
        },
        "reviews": [
            "Perfect for business trips",
            "Excellent workspace facilities",
            "Fast reliable WiFi"
        ],
        "description": "Premium business hotel with dedicated workspace areas. Located in the financial hub."
    }
]

# Generate 50 documents - FIXED VERSION
def generate_documents():
    documents = []  # Initialize empty list
    
    # Hotel description documents (1 per hotel)
    for hotel in hotels_data:
        # 1. Main description
        documents.append(f"HOTEL: {hotel['name']}\nDESCRIPTION: {hotel['description']}\nLOCATION: {hotel['location']}\nDISTANCE TO BEACH: {hotel['distance_to_beach']}\nRATING: {hotel['rating']}/5")
        
        # 2. Amenities document
        amenities_text = ", ".join(hotel["amenities"])
        documents.append(f"HOTEL: {hotel['name']}\nAMENITIES: {amenities_text}")
        
        # 3. Policies document
        policies = hotel["policies"]
        documents.append(f"HOTEL: {hotel['name']}\nPOLICIES:\n- Check-in: {policies['check_in']}\n- Check-out: {policies['check_out']}\n- Cancellation: {policies['cancellation']}\n- Pet Policy: {policies['pet_policy']}")
        
        # 4. Reviews document
        reviews_text = "\n".join([f"  - {review}" for review in hotel["reviews"][:3]])
        documents.append(f"HOTEL: {hotel['name']}\nGUEST REVIEWS:\n{reviews_text}")
        
        # 5. Location details
        documents.append(f"HOTEL: {hotel['name']}\nLOCATION DETAILS:\nAddress: {hotel['location']}\nNearby: Beach is {hotel['distance_to_beach']} away")
    
    # Add extra policy documents
    extra_policies = [
        "CANCELLATION POLICY STANDARD: Most hotels offer free cancellation. Specific terms vary by hotel. Always check individual hotel policies.",
        "CHECK-IN/CHECK-OUT STANDARD: Standard check-in is 2-4 PM. Standard check-out is 10 AM-12 PM. Early check-in may be available upon request.",
        "PET POLICY OVERVIEW: Pet policies vary. Some hotels allow pets with fees, others are pet-free except service animals.",
        "BREAKFAST INFORMATION: Many hotels offer complimentary breakfast. Types include continental, buffet, and made-to-order options.",
        "BEACH ACCESS: Hotels listed as 'near beach' are typically within 500 meters. Beach access may be public or private."
    ]
    
    documents.extend(extra_policies)
    
    return documents

# Generate documents
documents = generate_documents()

# Save raw documents
with open("data/raw_docs.txt", "w", encoding="utf-8") as f:
    for i, doc in enumerate(documents):
        f.write(f"=== DOCUMENT {i+1} ===\n{doc}\n\n")

# Save as JSON
with open("data/documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2)

# Save hotel metadata
with open("data/hotels_metadata.json", "w", encoding="utf-8") as f:
    json.dump(hotels_data, f, indent=2)

print(f"\n✅ Dataset generated successfully!")
print(f"📊 Statistics:")
print(f"   - Total documents: {len(documents)}")
print(f"   - Hotels covered: {len(hotels_data)}")
print(f"   - Files created:")
print(f"     • data/raw_docs.txt ({len(documents)} documents)")
print(f"     • data/documents.json")
print(f"     • data/hotels_metadata.json")

print(f"\n📄 Sample document (first 3):")
print("=" * 60)
for i in range(min(3, len(documents))):
    print(f"\n[Document {i+1}]\n{documents[i][:200]}...")
    print("-" * 40)

# Verify files exist
print("\n🔍 Verifying files...")
for file in ["data/raw_docs.txt", "data/documents.json", "data/hotels_metadata.json"]:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file} ({size} bytes)")
    else:
        print(f"   ❌ {file} not found")