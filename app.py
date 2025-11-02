"""
RAG Pipeline - Document Q&A Application
Main Streamlit application that orchestrates document ingestion,
processing, and question-answering using RAG.
"""

import streamlit as st

# Import configuration
from config import get_llm

# Import models
from models import get_embeddings

# Import services
from services import (
    DocumentIngestion,
    DocumentProcessor,
    VectorStoreManager,
    RetrievalService
)

# Import utilities
from utils import get_qa_prompt


# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the core components."""
    llm = get_llm()
    embeddings = get_embeddings()
    prompt = get_qa_prompt()
    processor = DocumentProcessor()
    retrieval_service = RetrievalService(llm, prompt)
    
    return llm, embeddings, prompt, processor, retrieval_service


# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []
    
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    
    if "vector_store_manager" not in st.session_state:
        _, embeddings, _, _, _ = initialize_components()
        st.session_state.vector_store_manager = VectorStoreManager(embeddings)


def create_vector_embeddings(uploaded_files, embeddings):
    """
    Create or update vector embeddings from uploaded files.
    
    Args:
        uploaded_files: List of uploaded file objects
        embeddings: Embeddings model instance
    """
    # Get current and previous file names
    current_file_names = [f.name for f in uploaded_files]
    previous_file_names = st.session_state.get("uploaded_file_names", [])
    
    # Find new files
    new_file_names, files_to_process = DocumentIngestion.get_new_files(
        uploaded_files,
        previous_file_names
    )
    
    # Check if there are new files to process
    if not new_file_names and st.session_state.vectors is not None:
        st.info("All files already embedded. No new documents to process.")
        return
    
    # Display processing info
  
    
    # Load documents
    all_docs, error_messages = DocumentIngestion.load_documents(files_to_process)
    
    # Display any errors
    for error in error_messages:
        st.warning(error)
    
    if not all_docs:
        st.warning("No documents were loaded.")
        return
    
    # Process documents
    _, _, _, processor, _ = initialize_components()
    documents = processor.split_documents(all_docs)
    
    # Create or update vector store
    vector_store_manager = st.session_state.vector_store_manager
    
    if st.session_state.vectors is None:
        # First time: create new vector store
        st.session_state.vectors = vector_store_manager.create_vector_store(documents)
        st.session_state.total_chunks = vector_store_manager.get_total_chunks()
    else:
        # Update existing vector store
        vector_store_manager.set_vector_store(
            st.session_state.vectors,
            st.session_state.total_chunks
        )
        st.session_state.vectors = vector_store_manager.update_vector_store(documents)
        st.session_state.total_chunks = vector_store_manager.get_total_chunks()
    
    # Update the list of processed files
    st.session_state.uploaded_file_names = current_file_names
    st.success("Embeddings created successfully.")


def process_query(user_query):
    """
    Process user query and display results.
    
    Args:
        user_query: User's question string
    """
    if st.session_state.vectors is None:
        st.warning("Please upload at least one PDF or TXT file to create embeddings first!")
        return
    
    with st.spinner("Searching for relevant information..."):
        _, _, _, _, retrieval_service = initialize_components()
        response = retrieval_service.process_query(user_query, st.session_state.vectors)
    
    # Display answer
    st.markdown("### Answer:")
    st.write(response['answer'])
    
    # Display source passages
    with st.expander("Document similarity search - View source passages"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"**Passage {i+1}:**")
            st.write(doc.page_content)
            st.write("---")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize components
    _, embeddings, _, _, _ = initialize_components()
    
    # Page configuration
    st.title("RAG Pipeline - Document Q&A")
    st.markdown("Ask questions about your documents using AI")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT files to create embeddings"
    )
    
    # Display upload status and auto-create embeddings
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        
        # Track which files are new
        current_files = [f.name for f in uploaded_files]
        previous_files = st.session_state.get("uploaded_file_names", [])
        new_files = [f for f in current_files if f not in previous_files]
        
        # Automatically create embeddings if there are new files
        if new_files or st.session_state.vectors is None:
            with st.spinner(f"Processing {len(new_files) if new_files else len(uploaded_files)} file(s) and creating embeddings... This may take a few moments."):
                create_vector_embeddings(uploaded_files, embeddings)
    
    # Query input section
    user_prompt = st.text_input("Enter your query about the documents")
    
    # Process query when entered
    if user_prompt:
        process_query(user_prompt)


if __name__ == "__main__":
    main()

