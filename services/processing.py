"""
Document processing service.
Handles text splitting and preprocessing.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MAX_DOCUMENTS_TO_PROCESS


class DocumentProcessor:
    """
    Service for processing documents.
    Splits documents into manageable chunks for embedding.
    """
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        Initialize the document processor.
        
        Args:
            chunk_size (int, optional): Size of each text chunk
            chunk_overlap (int, optional): Overlap between chunks
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def split_documents(self, documents, max_docs=None):
        """
        Split documents into chunks.
        
        Args:
            documents (list): List of documents to split
            max_docs (int, optional): Maximum number of documents to process
            
        Returns:
            list: List of document chunks
        """
        max_docs = max_docs or MAX_DOCUMENTS_TO_PROCESS
        docs_to_process = documents[:max_docs]
        
        chunks = self.text_splitter.split_documents(docs_to_process)
        return chunks
    
    def get_chunk_count(self, documents):
        """
        Get the total number of chunks that would be created.
        
        Args:
            documents (list): List of documents
            
        Returns:
            int: Number of chunks
        """
        chunks = self.split_documents(documents)
        return len(chunks)
