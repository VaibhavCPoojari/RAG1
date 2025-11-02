"""
Vector store service.
Manages vector store creation, updates, and persistence.
"""

from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Service for managing vector stores.
    Handles creation, updates, and retrieval operations.
    """
    
    def __init__(self, embeddings):
        """
        Initialize the vector store manager.
        
        Args:
            embeddings: Embeddings model instance
        """
        self.embeddings = embeddings
        self.vector_store = None
        self.total_chunks = 0
    
    def create_vector_store(self, documents):
        """
        Create a new vector store from documents.
        
        Args:
            documents (list): List of document chunks
            
        Returns:
            FAISS: Created vector store
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.total_chunks = len(documents)
        return self.vector_store
    
    def update_vector_store(self, documents):
        """
        Update existing vector store with new documents.
        
        Args:
            documents (list): List of new document chunks
            
        Returns:
            FAISS: Updated vector store
        """
        if self.vector_store is None:
            return self.create_vector_store(documents)
        
        # Create embeddings for new documents
        new_vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Merge with existing vector store
        self.vector_store.merge_from(new_vector_store)
        self.total_chunks += len(documents)
        
        return self.vector_store
    
    def get_vector_store(self):
        """
        Get the current vector store.
        
        Returns:
            FAISS: Current vector store
        """
        return self.vector_store
    
    def has_vector_store(self):
        """
        Check if a vector store exists.
        
        Returns:
            bool: True if vector store exists, False otherwise
        """
        return self.vector_store is not None
    
    def get_total_chunks(self):
        """
        Get the total number of chunks in the vector store.
        
        Returns:
            int: Total number of chunks
        """
        return self.total_chunks
    
    def set_vector_store(self, vector_store, total_chunks):
        """
        Set the vector store and chunk count from external source.
        
        Args:
            vector_store: FAISS vector store instance
            total_chunks (int): Total number of chunks
        """
        self.vector_store = vector_store
        self.total_chunks = total_chunks
