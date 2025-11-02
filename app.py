import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Import for Inference API
try:
    from huggingface_hub import InferenceClient
    HAS_INFERENCE_CLIENT = True
except ImportError:
    HAS_INFERENCE_CLIENT = False

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, groq_api_key=groq_api_key)

# Simple wrapper for HuggingFace Inference API that's compatible with FAISS
class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
        print(f" Using Hugging Face Inference API for embeddings (model: {model_name})")
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        print(f"Embedding {len(texts)} documents via HF Inference API...")
        embeddings = []
        # Process in batches to avoid timeout
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.client.feature_extraction(text, model=self.model_name)
                # feature_extraction returns a list (the embedding vector)
                embeddings.append(result)
        print(f"✓ Successfully embedded {len(embeddings)} documents")
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.client.feature_extraction(text, model=self.model_name)

# Choose embedding provider
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if huggingface_api_key and HAS_INFERENCE_CLIENT:
    emb = HuggingFaceInferenceEmbeddings(
        api_key=huggingface_api_key,
        model_name=HF_MODEL
    )
else:
    # Fallback to local embeddings
    print(f"✓ Using local HuggingFaceEmbeddings (model: {HF_MODEL})")
    emb = HuggingFaceEmbeddings(model_name=HF_MODEL)


 
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=emb
        st.session_state.loader=PyPDFDirectoryLoader("researchpapers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your Query from the research papers")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.success("Document Embedding Created Successfully")
    st.write("Your vector database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click 'Document Embedding' button first to create the vector database!")
    else:
        document_chain=create_stuff_documents_chain(llm, prompt)
        retriever=st.session_state.vectors.as_retriever()
        retriever_chain=create_retrieval_chain(retriever, document_chain)

        start=time.process_time()
        response=retriever_chain.invoke({'input':user_prompt})
        print(f"Response Time: {time.process_time() - start} seconds")
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-----------------------------")

