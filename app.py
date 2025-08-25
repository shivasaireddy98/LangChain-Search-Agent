import streamlit as st
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import traceback
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🔎 LangChain Search Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        border-left-color: #FF6B6B;
        background-color: #fff5f5;
    }
    
    .assistant-message {
        border-left-color: #4ECDC4;
        background-color: #f0fffe;
    }
    
    .tool-info {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        color: #c62828;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def validate_api_key(api_key):
    """Validate Groq API key format"""
    if not api_key:
        return False
    return len(api_key.strip()) > 20 and api_key.startswith('gsk_')

def initialize_tools():
    """Initialize and configure search tools"""
    try:
        # Wikipedia tool
        api_tool_wiki = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=2000
        )
        wiki = WikipediaQueryRun(
            api_wrapper=api_tool_wiki,
            name="wikipedia",
            description="Search Wikipedia for factual information, definitions, and general knowledge"
        )
        
        # ArXiv tool
        api_tool_arxiv = ArxivAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=2000
        )
        arxiv = ArxivQueryRun(
            api_wrapper=api_tool_arxiv,
            name="arxiv",
            description="Search ArXiv for academic papers and research articles"
        )
        
        # DuckDuckGo search tool
        search = DuckDuckGoSearchRun(
            name="web_search",
            description="Search the web for current information, news, and recent developments"
        )
        
        return [wiki, arxiv, search], None
    except Exception as e:
        return [], str(e)

def create_agent(api_key, tools):
    """Create and configure the search agent"""
    try:
        llm = ChatGroq(
            model_name="mixtral-8x7b-32768",  # Better model for complex queries
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        search_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        return search_agent, None
    except Exception as e:
        return None, str(e)

def format_response(response_text):
    """Format and clean up the response text"""
    if not response_text:
        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
    
    # Clean up common formatting issues
    response_text = re.sub(r'\n\s*\n\s*\n', '\n\n', response_text)  # Remove excessive newlines
    response_text = response_text.strip()
    
    return response_text

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant", 
                "content": "Hello! I'm your AI research assistant powered by LangChain. I can search Wikipedia, ArXiv, and the web to help you find accurate and up-to-date information. What would you like to know?"
            }
        ]
    
    if "conversation_count" not in st.session_state:
        st.session_state["conversation_count"] = 0

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<div class="main-header">🔎 LangChain Search Agent</div>', unsafe_allow_html=True)
    st.markdown("*Your intelligent research assistant with access to Wikipedia, ArXiv, and real-time web search*")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key", 
            type="password", 
            placeholder="gsk_...",
            help="Get your free API key from https://console.groq.com/",
            value=os.getenv("GROQ_API_KEY", "")
        )
        
        # API key validation
        if api_key:
            if validate_api_key(api_key):
                st.success("✅ Valid API key")
            else:
                st.error("❌ Invalid API key format")
        
        st.markdown("---")
        
        # Model selection
        model_option = st.selectbox(
            "🤖 Select Model",
            [
                "mixtral-8x7b-32768",
                "llama3-70b-8192", 
                "llama3-8b-8192",
                "gemma2-9b-it"
            ],
            help="Choose the AI model for processing your queries"
        )
        
        # Search tools info
        st.markdown("### 🛠️ Available Tools")
        tools, error = initialize_tools()
        
        if error:
            st.error(f"❌ Tool initialization error: {error}")
        else:
            for tool in tools:
                st.markdown(f"✅ **{tool.name}**: {tool.description}")
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.messages:
            st.markdown("### 📊 Session Stats")
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Queries", st.session_state.conversation_count)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
            st.session_state.conversation_count = 0
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What's the latest in quantum computing?",
            "Explain machine learning basics",
            "Recent developments in renewable energy",
            "What is the theory of relativity?",
            "Current AI research trends"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query[:10]}"):
                st.session_state.example_query = query
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle example query selection
        if hasattr(st.session_state, 'example_query'):
            prompt = st.session_state.example_query
            delattr(st.session_state, 'example_query')
        else:
            prompt = st.chat_input(
                placeholder="Ask me anything... (e.g., 'What are the latest developments in AI?')"
            )
        
        if prompt:
            # Validation
            if not api_key or not validate_api_key(api_key):
                st.error("❌ Please provide a valid Groq API key in the sidebar")
                return
            
            if not tools:
                st.error("❌ Search tools are not available. Please check your internet connection.")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🔍 Searching and analyzing..."):
                        # Create agent with selected model
                        llm = ChatGroq(
                            model_name=model_option,
                            groq_api_key=api_key,
                            temperature=0.1,
                            max_tokens=4096
                        )
                        
                        search_agent = initialize_agent(
                            tools=tools,
                            llm=llm,
                            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True,
                            verbose=True,
                            max_iterations=5,
                            early_stopping_method="generate"
                        )
                        
                        # Create callback handler
                        st_callback = StreamlitCallbackHandler(
                            st.container(), 
                            expand_new_thoughts=False,
                            collapse_completed_thoughts=True
                        )
                        
                        # Get response
                        response = search_agent.run(prompt, callbacks=[st_callback])
                        
                        # Format and display response
                        formatted_response = format_response(response)
                        st.markdown(formatted_response)
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": formatted_response
                        })
                        st.session_state.conversation_count += 1
                        
                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}"
                    st.error(error_message)
                    
                    # Show detailed error in expander
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                    # Add error to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error while processing your request. Please try again with a different question or check your API key."
                    })
    
    # Footer with tips
    st.markdown("---")
    with st.expander("📖 Usage Tips & Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Best Practices:**
            - Be specific in your questions
            - Ask for recent information when needed
            - Use follow-up questions for clarification
            - Try different phrasings if needed
            """)
        
        with col2:
            st.markdown("""
            **🔧 Features:**
            - Real-time web search via DuckDuckGo
            - Academic papers from ArXiv
            - Encyclopedia knowledge from Wikipedia
            - Intelligent source selection
            """)
        
        st.markdown("""
        **⚡ Models Available:**
        - **Mixtral-8x7b**: Best for complex research queries
        - **Llama3-70b**: High-quality responses, slower
        - **Llama3-8b**: Fast and efficient
        - **Gemma2-9b**: Balanced performance
        """)

if __name__ == "__main__":
    main()