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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import tempfile

 
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

 
class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
        
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
       
        embeddings = []
       
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.client.feature_extraction(text, model=self.model_name)
               
                embeddings.append(result)
       
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.client.feature_extraction(text, model=self.model_name)

 
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if huggingface_api_key and HAS_INFERENCE_CLIENT:
    emb = HuggingFaceInferenceEmbeddings(
        api_key=huggingface_api_key,
        model_name=HF_MODEL
    )
else:
  
    print(f" Using local HuggingFaceEmbeddings (model: {HF_MODEL})")
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

 
st.title(" RAG Pipeline - Document Q&A")
st.markdown("Ask questions about your documents using AI")
 
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Upload one or more PDF or TXT files to create embeddings"
)

if uploaded_files:
    st.success(f" {len(uploaded_files)} file(s) uploaded")
    # Track which files are new
    if "uploaded_file_names" in st.session_state:
        current_files = [f.name for f in uploaded_files]
        previous_files = st.session_state.uploaded_file_names
        new_files = [f for f in current_files if f not in previous_files]
        if new_files:
            st.info(f" {len(new_files)} new file(s) detected: {', '.join(new_files)}")

def create_vector_embeddings():
    st.session_state.embeddings = emb
    
    # Determine which files are new
    current_file_names = [f.name for f in uploaded_files]
    previous_file_names = st.session_state.get("uploaded_file_names", [])
    
    # Find new files
    new_file_names = [name for name in current_file_names if name not in previous_file_names]
    
    if not new_file_names and "vectors" in st.session_state:
        # No new files, vector store already exists
        st.info(" All files already embedded. No new documents to process.")
        return
    
    # Load only new documents
    files_to_process = [f for f in uploaded_files if f.name in new_file_names] if new_file_names else uploaded_files
    
    if new_file_names:
        st.info(f" Processing {len(files_to_process)} new file(s)...")
    else:
        st.info(f" Processing {len(files_to_process)} file(s)...")
    
    all_docs = []
    for uploaded_file in files_to_process:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()
            else:
                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                os.unlink(tmp_file_path)
                continue
            
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    if not all_docs:
        st.warning(" No documents were loaded.")
        return
    
 
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_documents = text_splitter.split_documents(all_docs[:50])
    
    
    # Create or append to vector store
    if "vectors" not in st.session_state:
        # First time: create new vector store
    
        st.session_state.vectors = FAISS.from_documents(new_documents, emb)
        st.session_state.total_chunks = len(new_documents)
    else:
        # Append to existing vector store
  
        # Create embeddings for new documents
        new_vector_store = FAISS.from_documents(new_documents, emb)
        # Merge with existing vector store
        st.session_state.vectors.merge_from(new_vector_store)
        st.session_state.total_chunks = st.session_state.get("total_chunks", 0) + len(new_documents)
    
    # Update the list of processed files
    st.session_state.uploaded_file_names = current_file_names
    st.success(f"Embeddings created successfully. ")

user_prompt = st.text_input(" Enter your query about the documents")

if st.button(" Create Document Embeddings"):
    if not uploaded_files:
        st.error(" Please upload at least one PDF or TXT file first!")
    else:
        with st.spinner("Creating embeddings... This may take a few moments."):
            create_vector_embeddings()

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning(" Please create document embeddings first by clicking the ' Create Document Embeddings' button!")
    else:
        with st.spinner("Searching for relevant information..."):
            document_chain=create_stuff_documents_chain(llm, prompt)
            retriever=st.session_state.vectors.as_retriever()
            retriever_chain=create_retrieval_chain(retriever, document_chain)

            start=time.process_time()
            response=retriever_chain.invoke({'input':user_prompt})
            response_time = time.process_time() - start
            
        print(f"Response Time: {response_time} seconds")
        
        st.markdown("###  Answer:")
        st.write(response['answer'])
        
       

        with st.expander(" Document similarity search - View source passages"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Passage {i+1}:**")
                st.write(doc.page_content)
                st.write("---")

