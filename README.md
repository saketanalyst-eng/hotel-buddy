# hotel-buddy
**Hotel RAG Q&A System**



# 🏨 Hotel RAG Q&A System

A smart hotel information system that answers your questions about hotels using real data. Built with RAG (Retrieval-Augmented Generation) - it finds relevant information from hotel documents and generates accurate answers.

## 🤔 What Does This System Do?

Have you ever struggled to find hotel information scattered across different pages? This system solves that problem. You can ask natural questions like:

- "Which hotels have free WiFi and breakfast?"
- "What's the cancellation policy for Seaside Resort?"
- "Suggest a hotel with good reviews near the beach"

The system reads through hotel documents, finds the most relevant information, and gives you a clear answer - all without hallucinating or making things up.

## 🛠️ How It Works (Simple Explanation)

Think of it like a smart librarian who:
1. **Reads all hotel documents** (descriptions, policies, reviews)
2. **Organizes information** into searchable chunks
3. **Finds relevant info** when you ask a question
4. **Answers using only** the found information (no guessing!)


## 📁 Project Structure


hotel-rag-system/
│
├── data/ # All your hotel data
│ ├── raw_docs.txt # Original hotel documents
│ ├── chunks.json # Text broken into pieces
│ ├── embeddings.npy # Vector representations
│ └── vectorizer.pkl # For converting text to vectors
│
├── src/ # Core system files
│ └── search_engine.py # Search functionality
│
├── outputs/ # Results from queries
│ ├── rag_results.json # Saved answers
│ └── evaluation_results.json # Performance metrics
│
├── step1_generate_dataset.py # Creates hotel data
├── step2_preprocessing.py # Cleans and chunks text
├── step3_embeddings.py # Creates vectors
├── step4_rag_system.py # Main RAG pipeline
├── rag_interactive.py # Chat with the system
├── evaluate_rag.py # Check performance

**How to use**

 Run Everything Automatically
   Run completesystem.py
   This will:
      *Load all hotel data
      *Set up the search engine
      *Run some example queries
      *Ask if you want to chat interactively
   option 2-interactive chat mode
    run python file python rag_interactive.py 
 **About data set**
  📊 What's In the Hotel Data?
     I created a dataset of 10 hotels with realistic information:

   📊 What's In the Hotel Data?
I created a dataset of 10 hotels with realistic information:

Hotel Name	Type	Key Features
Seaside Paradise Resort	Beachfront	Free WiFi, breakfast, pool, spa
City Central Hotel	Business	Free WiFi, business center, restaurant
Sunny Garden Inn	Budget	Free breakfast, garden, pet friendly
Luxury Grand Plaza	Luxury	Infinity pool, spa, premium service
Budget Stay Hostel	Budget	Shared kitchen, lockers, cheap
Ocean View Villas	Luxury	Private pool, beach access, kitchen
Airport Transit Hotel	Convenience	24/7 service, shuttle, soundproof
Mountain View Retreat	Nature	Hiking, bonfire, mountain views
Family Fun Resort	Family	Kids pool, playground, game room
Business Elite Suites	Premium	Meeting rooms, business lounge
Each hotel has:

Full description

Amenities list

Check-in/out policies

Cancellation rules

Pet policies

Guest reviews

**📈 Performance & Accuracy**
    
   For this you can run 
     evaluate_rag.py

**  Why Not 100%?**
   Some hotels have similar names or features
   Different phrasing of same amenity (e.g., "free internet" vs "WiFi included")
   Multiple policy types across documents

**🛡️ How We Prevent Hallucination (Making Things Up)****

       This is the most important part. The system NEVER invents information. Here's how:
           1. Strict Prompting
               The AI gets clear instructions: "ONLY use information from the provided context.                 If you don't know, say so."

           2. Low Temperature Setting
                I set temperature to 0.3 (range is 0-1). Lower = more factual, less creative.

           3. Empty Context Handling
               If no relevant info is found, the system says "I don't have that information"                     instead of guessing.

            4. Source Attribution
                Every answer must mention which hotel provided the information.

            5. Confidence Scoring
                  Each answer comes with a confidence score (0-1). Low confidence = not well                         supported.
  
 **📝 Limitations I Should Mention**
    Small dataset: Only 10 hotels (fine for demo, add more for production)

    Simple embeddings: TF-IDF works but misses semantic meaning

    No real-time updates: You'd need to re-run if hotel data changes

     Basic evaluation: More metrics like recall@k would be better

** ** 🎯 What I Learned Building This****
     Chunking matters: Breaking text into 500-character chunks with 20% overlap works best

     Hallucination is real: Without proper prompting, LLMs will invent hotel names

    Precision isn't perfect: Even good retrieval misses sometimes

    Ollama is great: Running local LLMs gives you privacy and control



**🚀 How to Extend This System**
   Want to make it better? Here are ideas:
       Add More Hotels
       Just add new hotels to hotels_data in step1_generate_dataset.py
       Use Better Embeddings
       Replace TF-IDF with Sentence-BERT (better semantic understanding):
       Add a Web Interface
       Use Streamlit to make a chat UI:
       Connect to Real Data
       Scrape hotel websites or use APIs (but check terms of service)
**📁 Files You'll Get After Running**
  File	                          What's Inside
data/raw_docs.txt	                 Original 55 hotel documents
data/chunks.json	                 Text split into searchable pieces
data/embeddings.npy                Vector numbers (machines understand)
data/vectorizer.pkl	               Tool to convert questions to vectors
outputs/rag_results.json	          Answers to test questions
outputs/evaluation_results.json    	Performance scores
**🙏 Credits & Acknowledgments**
    Ollama for making local LLMs easy
    Scikit-learn for vector search tools
    FAISS (optional) for fast similarity search
**📄 License**
    This project is for learning purposes. Feel free to use, modify, and share.

    

**💬 Questions or Issues?**
   If something doesn't work:

   Check the troubleshooting section above

    Make sure all packages are installed

    Verify Ollama is running (if using)

    Run python test_system.py to diagnose
  
