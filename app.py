import streamlit as st
from typing import List, Dict, Any, Optional, Literal
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import json
import os
from datetime import datetime
import traceback
from pathlib import Path
import pickle

# Page configuration
st.set_page_config(
    page_title="🧠 LangGraph Smart Router",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .route-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .local-route {
        background: linear-gradient(45deg, #4CAF50, #45a049);
    }
    
    .external-route {
        background: linear-gradient(45deg, #2196F3, #1976d2);
    }
    
    .hybrid-route {
        background: linear-gradient(45deg, #FF9800, #f57c00);
    }
    
    .graph-node {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .active-node {
        border-color: #007bff;
        background: #e7f3ff;
    }
    
    .step-indicator {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the state for our graph
class GraphState(TypedDict):
    query: str
    route_decision: Optional[str]
    local_results: Optional[List[Dict]]
    external_results: Optional[str]
    final_answer: Optional[str]
    context: Optional[str]
    step_history: List[str]

class LocalDataManager:
    """Manages local knowledge base with vector search"""
    
    def __init__(self):
        self.vector_store = None
        self.documents = []
        self.embeddings = None
        
    def initialize_embeddings(self, openai_api_key: str = None):
        """Initialize embeddings - try OpenAI first, then fallback"""
        try:
            if openai_api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            else:
                # Fallback to a free alternative (you might want to use a local model)
                st.warning("Using basic text similarity (OpenAI embeddings not available)")
                return False
            return True
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def load_sample_data(self):
        """Load sample knowledge base"""
        sample_docs = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
                "metadata": {"topic": "programming", "language": "python"}
            },
            {
                "content": "Machine Learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. Common algorithms include linear regression, decision trees, and neural networks.",
                "metadata": {"topic": "machine_learning", "category": "ai"}
            },
            {
                "content": "Streamlit is an open-source app framework for Machine Learning and Data Science projects. It allows you to create web apps with simple Python scripts.",
                "metadata": {"topic": "programming", "framework": "streamlit"}
            },
            {
                "content": "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
                "metadata": {"topic": "ai", "framework": "langchain"}
            },
            {
                "content": "Vector databases store high-dimensional vectors and enable similarity search. They are essential for RAG (Retrieval Augmented Generation) applications.",
                "metadata": {"topic": "databases", "category": "vector"}
            }
        ]
        
        self.documents = [
            Document(page_content=doc["content"], metadata=doc["metadata"]) 
            for doc in sample_docs
        ]
        
        return len(self.documents)
    
    def create_vector_store(self):
        """Create vector store from documents"""
        if not self.embeddings or not self.documents:
            return False
            
        try:
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            return True
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return False
    
    def search_local(self, query: str, k: int = 3) -> List[Dict]:
        """Search local knowledge base"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
        except Exception as e:
            st.error(f"Local search error: {e}")
            return []

class ExternalTools:
    """Manages external search tools"""
    
    def __init__(self):
        self.tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize external search tools"""
        try:
            # Wikipedia tool
            wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
            self.tools['wikipedia'] = WikipediaQueryRun(api_wrapper=wiki_wrapper)
            
            # ArXiv tool
            arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
            self.tools['arxiv'] = ArxivQueryRun(api_wrapper=arxiv_wrapper)
            
            # Web search tool
            self.tools['web_search'] = DuckDuckGoSearchRun()
            
        except Exception as e:
            st.error(f"Failed to initialize external tools: {e}")
    
    def search_external(self, query: str, tool_name: str) -> str:
        """Search using external tools"""
        if tool_name not in self.tools:
            return f"Tool {tool_name} not available"
        
        try:
            result = self.tools[tool_name].run(query)
            return result
        except Exception as e:
            return f"External search error: {str(e)}"

class LangGraphRouter:
    """Main LangGraph router class"""
    
    def __init__(self, groq_api_key: str, openai_api_key: str = None):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1
        )
        self.local_manager = LocalDataManager()
        self.external_tools = ExternalTools()
        self.graph = None
        
        # Initialize local data
        if openai_api_key:
            if self.local_manager.initialize_embeddings(openai_api_key):
                self.local_manager.load_sample_data()
                self.local_manager.create_vector_store()
    
    def route_query(self, state: GraphState) -> GraphState:
        """Decide whether to use local search, external tools, or both"""
        
        query = state["query"]
        
        # Router prompt
        router_prompt = PromptTemplate(
            template="""
            You are a routing system that decides how to answer user queries.
            
            Query: {query}
            
            Based on the query, decide the best approach:
            - "local": If the query is about programming, AI/ML basics, or technical concepts that might be in our knowledge base
            - "external": If the query needs current information, specific facts, or recent news
            - "hybrid": If the query would benefit from both local knowledge and external information
            
            Available local topics: Python, Machine Learning, Streamlit, LangChain, Vector Databases
            
            Respond with only one word: local, external, or hybrid
            """,
            input_variables=["query"]
        )
        
        try:
            response = self.llm.invoke(router_prompt.format(query=query))
            route = response.content.strip().lower()
            
            if route not in ["local", "external", "hybrid"]:
                route = "hybrid"  # Default fallback
            
            state["route_decision"] = route
            state["step_history"].append(f"🔄 Routed to: {route}")
            
        except Exception as e:
            state["route_decision"] = "hybrid"
            state["step_history"].append(f"❌ Router error, using hybrid: {str(e)}")
        
        return state
    
    def search_local_node(self, state: GraphState) -> GraphState:
        """Search local knowledge base"""
        
        if state["route_decision"] not in ["local", "hybrid"]:
            return state
        
        query = state["query"]
        results = self.local_manager.search_local(query, k=3)
        
        state["local_results"] = results
        
        if results:
            state["step_history"].append(f"📚 Found {len(results)} local results")
        else:
            state["step_history"].append("📚 No local results found")
        
        return state
    
    def search_external_node(self, state: GraphState) -> GraphState:
        """Search external sources"""
        
        if state["route_decision"] not in ["external", "hybrid"]:
            return state
        
        query = state["query"]
        
        # Determine best external tool based on query
        if "research" in query.lower() or "paper" in query.lower():
            tool = "arxiv"
        elif "current" in query.lower() or "news" in query.lower() or "recent" in query.lower():
            tool = "web_search"
        else:
            tool = "wikipedia"
        
        result = self.external_tools.search_external(query, tool)
        state["external_results"] = result
        state["step_history"].append(f"🌐 Searched {tool}")
        
        return state
    
    def synthesize_answer(self, state: GraphState) -> GraphState:
        """Combine results and generate final answer"""
        
        query = state["query"]
        local_results = state.get("local_results", [])
        external_results = state.get("external_results", "")
        
        # Build context
        context_parts = []
        
        if local_results:
            local_context = "\n".join([f"Local: {result['content']}" for result in local_results])
            context_parts.append(f"LOCAL KNOWLEDGE:\n{local_context}")
        
        if external_results:
            context_parts.append(f"EXTERNAL INFORMATION:\n{external_results}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific information found."
        state["context"] = context
        
        # Generate final answer
        synthesis_prompt = PromptTemplate(
            template="""
            Based on the following information, provide a comprehensive and accurate answer to the user's question.
            
            Question: {query}
            
            Available Information:
            {context}
            
            Instructions:
            1. Synthesize information from all sources
            2. Clearly indicate if information is from local knowledge vs external sources
            3. If no relevant information is found, say so honestly
            4. Provide a helpful and informative response
            
            Answer:
            """,
            input_variables=["query", "context"]
        )
        
        try:
            response = self.llm.invoke(synthesis_prompt.format(query=query, context=context))
            state["final_answer"] = response.content
            state["step_history"].append("✅ Generated final answer")
            
        except Exception as e:
            state["final_answer"] = f"I apologize, but I encountered an error while generating the response: {str(e)}"
            state["step_history"].append(f"❌ Synthesis error: {str(e)}")
        
        return state
    
    def create_graph(self):
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("route", self.route_query)
        workflow.add_node("local_search", self.search_local_node)
        workflow.add_node("external_search", self.search_external_node)
        workflow.add_node("synthesize", self.synthesize_answer)
        
        # Add edges
        workflow.set_entry_point("route")
        workflow.add_edge("route", "local_search")
        workflow.add_edge("route", "external_search")
        workflow.add_edge("local_search", "synthesize")
        workflow.add_edge("external_search", "synthesize")
        workflow.add_edge("synthesize", END)
        
        self.graph = workflow.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the graph"""
        
        if not self.graph:
            self.create_graph()
        
        # Initialize state
        initial_state: GraphState = {
            "query": query,
            "route_decision": None,
            "local_results": None,
            "external_results": None,
            "final_answer": None,
            "context": None,
            "step_history": []
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            return final_state
            
        except Exception as e:
            return {
                "query": query,
                "route_decision": "error",
                "local_results": None,
                "external_results": None,
                "final_answer": f"An error occurred: {str(e)}",
                "context": None,
                "step_history": [f"❌ Graph execution error: {str(e)}"]
            }

def main():
    # Header
    st.markdown('<div class="main-header">🧠 LangGraph Smart Router</div>', unsafe_allow_html=True)
    st.markdown("*Intelligent query routing with local knowledge base and external tools*")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Keys
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Required for LLM routing and synthesis"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            placeholder="sk-...",
            help="For better local embeddings. If not provided, external tools only."
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 🔧 System Status")
        
        if groq_api_key:
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Key Required")
        
        if openai_api_key:
            st.success("✅ Local Search Available")
        else:
            st.warning("⚠️ Local Search Disabled")
        
        st.markdown("---")
        
        # Sample Data Info
        with st.expander("📚 Local Knowledge Base"):
            st.markdown("""
            **Available Topics:**
            - Python Programming
            - Machine Learning Basics
            - Streamlit Framework
            - LangChain Framework  
            - Vector Databases
            
            **Sample Questions:**
            - "What is Python?"
            - "Explain machine learning"
            - "How does Streamlit work?"
            """)
        
        # Clear chat
        if st.button("🗑️ Clear Chat", type="secondary"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your intelligent assistant with access to both local knowledge and external tools. Ask me anything!"
            }
        ]
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 💡 Example Queries")
        
        examples = [
            "What is Python programming?",
            "Latest AI research papers",
            "Current news about technology",
            "Explain machine learning",
            "How does vector search work?"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example[:15]}"):
                st.session_state.example_query = example
    
    with col1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show route info for assistant messages
                if message["role"] == "assistant" and "route_info" in message:
                    route = message["route_info"]["route_decision"]
                    if route:
                        route_class = f"{route}-route"
                        st.markdown(
                            f'<div class="route-indicator {route_class}">Route: {route.title()}</div>',
                            unsafe_allow_html=True
                        )
        
        # Handle example selection
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            delattr(st.session_state, 'example_query')
        else:
            query = st.chat_input("Ask me anything...")
        
        if query:
            # Validate API key
            if not groq_api_key:
                st.error("❌ Please provide your Groq API key in the sidebar")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Process query
            with st.chat_message("assistant"):
                with st.spinner("🧠 Processing query through LangGraph..."):
                    
                    # Initialize router
                    router = LangGraphRouter(groq_api_key, openai_api_key)
                    
                    # Process query
                    result = router.process_query(query)
                    
                    # Show processing steps
                    with st.expander("🔍 Processing Steps", expanded=False):
                        for step in result.get("step_history", []):
                            st.markdown(f"<span class='step-indicator'>{step}</span>", unsafe_allow_html=True)
                    
                    # Display route decision
                    route = result.get("route_decision", "unknown")
                    route_class = f"{route}-route"
                    st.markdown(
                        f'<div class="route-indicator {route_class}">🎯 Route Decision: {route.title()}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display results breakdown
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        local_results = result.get("local_results", [])
                        if local_results:
                            st.markdown("**📚 Local Knowledge Found:**")
                            for i, res in enumerate(local_results[:2]):
                                st.markdown(f"• {res['content'][:100]}...")
                    
                    with col_b:
                        external_results = result.get("external_results")
                        if external_results:
                            st.markdown("**🌐 External Information:**")
                            st.markdown(f"• {external_results[:100]}...")
                    
                    # Display final answer
                    final_answer = result.get("final_answer", "No answer generated")
                    st.markdown("**🎯 Final Answer:**")
                    st.markdown(final_answer)
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "route_info": result
                    })
    
    # Footer
    st.markdown("---")
    with st.expander("📖 How It Works"):
        st.markdown("""
        ### 🧠 LangGraph Workflow
        
        1. **🔄 Router Node**: LLM decides whether to use local knowledge, external tools, or both
        2. **📚 Local Search Node**: Searches vector database of local knowledge
        3. **🌐 External Search Node**: Uses Wikipedia, ArXiv, or web search
        4. **🎯 Synthesis Node**: Combines all information into a comprehensive answer
        
        ### 🛠️ Available Tools
        - **Local Vector Search**: Embedded knowledge base with similarity search
        - **Wikipedia**: General knowledge and definitions
        - **ArXiv**: Academic papers and research
        - **DuckDuckGo**: Current web information
        
        ### 🚀 Benefits
        - **Intelligent Routing**: Automatically chooses best information sources
        - **Hybrid Approach**: Combines local knowledge with external tools
        - **Transparent Process**: See exactly how your query is processed
        - **Extensible**: Easy to add new tools and knowledge sources
        """)

if __name__ == "__main__":
    main()