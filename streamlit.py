import os
import streamlit as st
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define system prompt (you'll need to add your actual system prompt here)
system_prompt = """You are a helpful and knowledgeable assistant specializing in answering questions about the ChaiCode documentation from https://docs.chaicode.com/

## Your Role
You provide accurate, detailed, and practical answers based on the provided documentation chunks. Your responses should be educational, well-structured, and include relevant code examples when applicable.

## Response Guidelines

### Structure & Content
- Analyze the provided documentation chunks carefully
- Provide comprehensive answers that directly address the user's question
- Include relevant code examples, configurations, or commands ONLY WHEN PRESENT IN THE DOCUMENTATION
- Explain concepts clearly with step-by-step instructions when appropriate
- ALWAYS include relevant URLs from the documentation to help users access the full content

### URL Integration
- Include specific section URLs (with anchors) when referencing particular topics
- Format URLs naturally within your response for easy access
- Use phrases like "You can find more details at [URL]" or "For the complete guide, visit [URL]"

### Code Examples
- Present code examples in properly formatted code blocks with language specification
- Explain code line-by-line when the documentation provides such explanations
- Include file names when specified in the documentation

### Technical Accuracy
- Maintain technical precision when discussing programming concepts
- Include specific version numbers, package names, and installation commands as provided
- Preserve exact syntax and naming conventions from the documentation

### Suggest Relevant Follow-up Questions
- Suggest some relevant follow-up questions to the user depending on their current question and the context provided.

## Example Interactions for your reference
---
**Q: What are the primitive data types in C++ according to ChaiCode documentation?**

A: According to the ChaiCode C++ documentation, primitive data types are the basic data types used to define what type of data a variable can hold. The main primitive data types in C++ include:

**Core Primitive Data Types:**
- `int`: Signed integer data type (values from -2,147,483,648 to 2,147,483,647)
- `float`: Floating-point data type with 6 decimal places precision
- `double`: Floating-point data type with 15 decimal places precision  
- `char`: Character data type that holds a single character
- `bool`: Boolean data type that holds either `true` or `false`
- `void`: Used to indicate that a function doesn't return a value

**Additional Primitive Data Types:**
- `short`: Signed integer (-32,768 to 32,767)
- `long`: Signed integer (-2,147,483,648 to 2,147,483,647)
- `long long`: Signed integer (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
- Various unsigned types like `unsigned char`, `unsigned short`, `unsigned int`, etc.

Here's a practical example from the documentation:

```cpp
#include<iostream>
using namespace std;
int main(){
    int teaLeaves = 50;
    float waterTemperature = 85.588588;
    double priceOfTea = 299.99;
    char teaGrade = 'A';
    bool isTeaReady = false;
    cout << waterTemperature << endl;
    return 0;
}

**Suggested follow-up questions:**
- What are operators in C++?
- What is the difference between variables and constants in C++?
- What are the different types of loops in C++?
---
Now please analyze the user query with respect to the given context, and answer the question in a detailed manner by only using the context provided.
If user query is not related to the context, then say "I'm sorry, I could not find relevant information in the documentation to answer your question. Please visit https://docs.chaicode.com/ to get more information."
"""

@st.cache_resource
def initialize_components():
    """Initialize embeddings, vector store, and LLM model"""
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024
    )
    
    # Initialize the qdrant vector store
    EMBEDDING_VECTOR_SIZE = 1024
    qdrant = QdrantVectorStore(
        client=QdrantClient(
            url=os.getenv("qdrant_endpoint"),
            api_key=os.getenv("qdrant_api_key"),
            https=False,
            timeout=600,
        ),
        collection_name="chai-docs",
        embedding=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBEDDING_VECTOR_SIZE),
    )
    
    # Initialize LLM model
    llm_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )
    
    return embeddings, qdrant, llm_model

def process_question(question, embeddings, qdrant, llm_model, conversation_history):
    """Process the user question and return streaming response"""
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Converting query to vector embedding
        status_placeholder = st.empty()
        status_placeholder.info("🔄 Converting your query to vector embedding...")
        time.sleep(0.5)  # Small delay for better UX
        
        question_embedding = embeddings.embed_query(question)
        
        # Step 2: Finding relevant chunks
        status_placeholder.info("🔍 Searching for relevant information in the knowledge base...")
        time.sleep(0.5)
        
        relevant_chunks = qdrant.similarity_search(question, k=5)
        relevant_chunks_str = "\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Step 3: Generating response
        status_placeholder.info("✨ Generating your personalized response...")
        time.sleep(0.5)
        
        # Clear status
        status_placeholder.empty()
    
    # Prepare LLM input with conversation history
    llm_input = [SystemMessage(content=system_prompt)]
    
    # Add conversation history (limit to last 10 messages to avoid token limits)
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    for msg in recent_history:
        if msg["role"] == "user":
            llm_input.append(HumanMessage(content=msg["content"]))
        else:
            llm_input.append(AIMessage(content=msg["content"]))
    
    # Add current question with relevant chunks
    current_query = f"Question: {question}\n\nRelevant chunks from knowledge base: {relevant_chunks_str}"
    llm_input.append(HumanMessage(content=current_query))
    
    # Stream the response
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        for chunk in llm_model.stream(llm_input):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌") #typing effect
        
        # Final response without cursor
        response_placeholder.markdown(full_response)
        
        # Show relevant chunks in expander
        with st.expander("📚 View Retrieved data", expanded=False):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Source {i}:**")
                if chunk.metadata:
                    st.json(chunk.metadata)
                st.markdown("---")
        
        return full_response
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat with Chai-docs",
        page_icon="🍵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Starter question buttons styling */
    div[data-testid="column"] .stButton > button {
        height: 60px;
        font-size: 14px;
        text-align: left;
        white-space: normal;
        word-wrap: break-word;
        padding: 10px 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍵 Chat with Chai-docs</h1>
        <p>Ask me anything about Chai documentation and I'll help you find the answers!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    try:
        embeddings, qdrant, llm_model = initialize_components()
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "process_starter_question" not in st.session_state:
        st.session_state.process_starter_question = None
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Show starter questions only if no messages yet
    if not st.session_state.messages:
        st.markdown("### 🚀 Quick Start - Try these questions:")
        
        # Define starter questions
        starter_questions = [
            "What are the primitive data types in C++?",
            "How to add css and js to django project?",
        ]
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Display starter questions as clickable buttons
        for i, question in enumerate(starter_questions):
            if i % 2 == 0:
                with col1:
                    if st.button(f"💡 {question}", key=f"starter_{i}", use_container_width=True):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": question})
                        # Set flag to process this question
                        st.session_state.process_starter_question = question
                        st.rerun()
            else:
                with col2:
                    if st.button(f"💡 {question}", key=f"starter_{i}", use_container_width=True):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": question})
                        # Set flag to process this question
                        st.session_state.process_starter_question = question
                        st.rerun()
        
        st.markdown("---")
        st.markdown("**Or type your own question below:**")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process starter question if one was clicked
    if hasattr(st.session_state, 'process_starter_question') and st.session_state.process_starter_question:
        question = st.session_state.process_starter_question
        # Clear the flag
        st.session_state.process_starter_question = None
        
        # Display assistant response
        with st.chat_message("assistant"):
            response = process_question(question, embeddings, qdrant, llm_model, st.session_state.messages)
            if response:
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat input
    if prompt := st.chat_input("Ask your question about Chai-docs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            response = process_question(prompt, embeddings, qdrant, llm_model, st.session_state.messages)
            if response:
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### 🔧 About")
        st.markdown("""
        This chat application uses:
        - **OpenAI Embeddings** for semantic search
        - **Qdrant Vector Store** for document retrieval
        - **GPT-4.1** for intelligent responses
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about Chai documentation topics
        - Ask follow-up questions for clarification
        - Visit original documentation from provided links for more information
        """)
        
        # Show conversation stats
        if st.session_state.messages:
            st.markdown("### 📊 Session Stats")
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.markdown(f"**Questions asked:** {user_messages}")
            st.markdown(f"**Total messages:** {len(st.session_state.messages)}")
        
        st.markdown("---")
        st.markdown("### 🔄 Follow-up Questions")
        st.markdown("""
        You can ask follow-up questions that reference:
        - Previous answers
        - Earlier topics in the conversation
        - Clarifications or deeper explanations
        
        *Example: "Can you explain that last point in more detail?" or "How does this relate to what you mentioned earlier?"*
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()