"""
Custom embedding models for the RAG pipeline.
Provides both API-based and local embedding options.
"""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import HUGGINGFACE_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


class HuggingFaceInferenceEmbeddings(Embeddings):
    """
    Custom embeddings class using HuggingFace Inference API.
    Provides document and query embedding capabilities.
    """
    
    def __init__(self, api_key, model_name):
        """
        Initialize the HuggingFace Inference embeddings.
        
        Args:
            api_key (str): HuggingFace API key
            model_name (str): Name of the model to use for embeddings
        """
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
        
    def embed_documents(self, texts):
        """
        Embed a list of documents.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            list: List of embeddings
        """
        embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.client.feature_extraction(text, model=self.model_name)
                embeddings.append(result)
        
        return embeddings
    
    def embed_query(self, text):
        """
        Embed a single query.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            list: Embedding vector
        """
        return self.client.feature_extraction(text, model=self.model_name)


def get_embeddings():
    """
    Get the appropriate embeddings model based on availability.
    
    Returns:
        Embeddings: An instance of embeddings model (API-based or local)
    """
 
    print(f"Using HuggingFace Inference API (model: {EMBEDDING_MODEL_NAME})")
    return HuggingFaceInferenceEmbeddings(
            api_key=HUGGINGFACE_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
 
