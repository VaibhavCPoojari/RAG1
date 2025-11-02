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
        print(f" Using Hugging Face Inference API for embeddings (model: {model_name})")
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        print(f"Embedding {len(texts)} documents via HF Inference API...")
        embeddings = []
       
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.client.feature_extraction(text, model=self.model_name)
               
                embeddings.append(result)
        print(f"✓ Successfully embedded {len(embeddings)} documents")
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

 
st.title(" RAG Pipeline - Document Q&A")
st.markdown("Ask questions about your documents using AI")
 
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Upload one or more PDF or TXT files to create embeddings"
)

if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded")

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=emb
         
        st.info("Loading documents from uploaded files...")
        all_docs = []
        for uploaded_file in uploaded_files:
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
        
        st.session_state.docs = all_docs
        st.info(f" Loaded {len(all_docs)} pages/documents from {len(uploaded_files)} file(s)")
        
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.info(f" Split into {len(st.session_state.documents)} chunks")
        st.session_state.vectors=FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

user_prompt = st.text_input(" Enter your query about the documents")

if st.button("Create Document Embeddings"):
    if not uploaded_files:
        st.error(" Please upload at least one PDF or TXT file first!")
    else:
        with st.spinner("Creating embeddings... This may take a few moments."):
            create_vector_embeddings()
        st.success(" Document Embedding Created Successfully")
        st.write("Your vector database is ready")

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

