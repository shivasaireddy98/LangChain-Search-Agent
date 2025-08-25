# streamlit-app
🧠 LangGraph Smart Router
An intelligent query routing system built with LangGraph that intelligently decides whether to search local knowledge base or external tools, then synthesizes comprehensive answers.
Show Image
Show Image
Show Image
✨ Features
🎯 Intelligent Routing

LLM-Powered Router: Automatically decides the best information sources
Local-First Strategy: Searches local knowledge base before external tools
Hybrid Approach: Combines multiple sources when beneficial
Transparent Process: Visual workflow showing decision-making steps

📚 Local Knowledge Management

Vector Search: FAISS-powered semantic search
Embeddings: OpenAI embeddings for high-quality similarity matching
Sample Knowledge Base: Pre-loaded with programming and AI topics
Extensible: Easy to add your own documents

🌐 External Tools Integration

Wikipedia: General knowledge and definitions
ArXiv: Academic papers and research articles
DuckDuckGo: Real-time web search for current information
Tool Selection: Intelligent tool selection based on query context

🔄 LangGraph Workflow
Query → Router → Local Search ⟊
                           ⟶ Synthesizer → Final Answer
            → External Search ⟊
🏗️ Architecture
Graph Nodes

Router Node: LLM analyzes query and decides routing strategy
Local Search Node: Searches vector database with semantic similarity
External Search Node: Queries appropriate external tools
Synthesis Node: Combines all information into comprehensive answer

State Management

GraphState: TypedDict managing query flow state
Step History: Tracks processing steps for transparency
Context Preservation: Maintains information across nodes

🚀 Quick Start
Prerequisites

Python 3.8+
Groq API key (free at console.groq.com)
OpenAI API key (optional, for local embeddings)

Installation
bash# Clone repository
git clone https://github.com/yourusername/langgraph-smart-router.git
cd langgraph-smart-router

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
⚙️ Configuration
API Keys Setup
Create a .env file:
envGROQ_API_KEY=gsk_your_groq_api_key_here
OPENAI_API_KEY=sk_your_openai_key_here  # Optional
Or enter them in the Streamlit sidebar.
Local Knowledge Base
The app includes a sample knowledge base with topics:

Python Programming
Machine Learning Basics
Streamlit Framework
LangChain Framework
Vector Databases

Adding Your Own Data
python# Add documents to local knowledge base
sample_docs = [
    {
        "content": "Your knowledge content here...",
        "metadata": {"topic": "your_topic", "source": "your_source"}
    }
]
📖 Usage Examples
Basic Queries
"What is Python programming?"
→ Route: Local (finds programming knowledge)
→ Result: Detailed Python explanation from local KB

"Latest AI research papers"
→ Route: External (ArXiv search)
→ Result: Recent research papers from ArXiv

"Current news about technology"
→ Route: External (Web search)
→ Result: Latest tech news from DuckDuckGo
Hybrid Queries
"Explain machine learning and recent developments"
→ Route: Hybrid
→ Local: ML fundamentals from knowledge base
→ External: Recent developments from web/ArXiv
→ Synthesis: Combined comprehensive answer
🛠️ Customization
Adding New Tools
pythonclass CustomTool:
    def search(self, query: str) -> str:
        # Your custom search logic
        return "Custom search results"

# Add to external tools
external_tools.tools['custom'] = CustomTool()
Modifying Router Logic
pythondef custom_route_query(self, state: GraphState) -> GraphState:
    # Custom routing logic
    query = state["query"]
    
    # Your decision logic here
    if "custom_condition" in query:
        route = "custom_route"
    else:
        route = "default"
    
    state["route_decision"] = route
    return state
Extending Local Knowledge
python# Load documents from files
def load_documents_from_folder(folder_path: str):
    documents = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, 'r') as f:
            content = f.read()
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path.name}
            ))
    return documents
🧪 Testing
Sample Queries
Local Knowledge:

"What is Streamlit?"
"Explain vector databases"
"How does machine learning work?"

External Search:

"Current AI trends 2024"
"Recent papers on quantum computing"
"Latest Python updates"

Hybrid Approach:

"Python machine learning libraries and recent developments"
"Explain LangChain and its latest features"

🔧 Troubleshooting
Common Issues
"No local results found"

Check if OpenAI API key is provided
Verify vector store initialization
Try more general queries

"External search error"

Check internet connection
Verify external tool APIs are working
Try different query phrasing

"Router decision error"

Check Groq API key validity
Verify model availability
Review router prompt logic

Performance Optimization
python# Optimize vector search
vector_store = FAISS.from_documents(
    documents, 
    embeddings,
    distance_strategy=DistanceStrategy.COSINE  # Better for text
)

# Optimize LLM calls
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192",  # Faster model
    temperature=0.0  # More deterministic
)