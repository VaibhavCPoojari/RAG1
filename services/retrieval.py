"""
Retrieval service.
Handles query processing and response generation.
"""

import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class RetrievalService:
    """
    Service for handling document retrieval and query processing.
    """
    
    def __init__(self, llm, prompt_template):
        """
        Initialize the retrieval service.
        
        Args:
            llm: Language model instance
            prompt_template: Prompt template for queries
        """
        self.llm = llm
        self.prompt_template = prompt_template
    
    def create_retrieval_chain(self, vector_store):
        """
        Create a retrieval chain from vector store.
        
        Args:
            vector_store: FAISS vector store instance
            
        Returns:
            tuple: (retrieval_chain, document_chain, retriever)
        """
        document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, document_chain, retriever
    
    def process_query(self, query, vector_store):
        """
        Process a user query and return the response.
        
        Args:
            query (str): User query
            vector_store: FAISS vector store instance
            
        Returns:
            dict: Response containing answer, context, and metadata
        """
        # Create retrieval chain
        retrieval_chain, _, _ = self.create_retrieval_chain(vector_store)
        
        # Measure response time
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        response_time = time.process_time() - start_time
        
        # Add response time to the result
        response['response_time'] = response_time
        
        print(f"Response Time: {response_time} seconds")
        
        return response
    
    def get_relevant_documents(self, query, vector_store, k=4):
        """
        Get relevant documents for a query without generating an answer.
        
        Args:
            query (str): User query
            vector_store: FAISS vector store instance
            k (int): Number of documents to retrieve
            
        Returns:
            list: List of relevant documents
        """
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return docs
