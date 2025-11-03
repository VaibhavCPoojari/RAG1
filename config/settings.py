"""
Configuration settings for the RAG pipeline application.
Manages environment variables, API keys, and model initialization.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Set environment variables
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY if HUGGINGFACE_API_KEY else ""

# Model Configuration
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_DOCUMENTS_TO_PROCESS = 50

# Batch Configuration
EMBEDDING_BATCH_SIZE = 10


def get_llm():
    """
    Initialize and return the LLM instance.
    
    Returns:
        ChatGroq: Initialized language model
    """
    return ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY
    )


 
