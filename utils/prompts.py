"""
Prompt templates for the RAG pipeline.
Defines the structure and format of prompts used for querying.
"""

from langchain_core.prompts import ChatPromptTemplate


def get_qa_prompt():
    """
    Get the question-answering prompt template.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for Q&A
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    return prompt
