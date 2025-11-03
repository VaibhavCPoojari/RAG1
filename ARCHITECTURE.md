# RAG Pipeline - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Design Patterns](#design-patterns)
4. [Component Architecture](#component-architecture)
5. [Data Flow](#data-flow)
6. [Technology Stack](#technology-stack)
7. [Scalability Considerations](#scalability-considerations)

---

## System Overview

### Purpose
The RAG (Retrieval-Augmented Generation) Pipeline is a document question-answering system that combines document retrieval with Large Language Model (LLM) generation to provide accurate, context-grounded answers to user queries.

### Key Features
- **Multi-format Document Support**: PDF and TXT files
- **Semantic Search**: Vector-based similarity search using embeddings
- **Incremental Updates**: Efficient processing of new documents without reprocessing existing ones
- **Source Citations**: Transparent answer generation with source document references
- **Real-time Processing**: Automatic embedding generation on file upload
- **Session Persistence**: Maintains vector store across user interactions

### Architecture Philosophy
The project follows **Clean Architecture** principles with clear separation of concerns:
- **Modular Design**: Each module has a single, well-defined responsibility
- **Dependency Inversion**: High-level modules don't depend on low-level implementation details
- **Testability**: Components can be tested independently
- **Extensibility**: Easy to add new features without modifying existing code

---

## Architecture Diagram

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                      (Streamlit Web App)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│                         (app.py)                                 │
│  • Session State Management                                      │
│  • UI Orchestration                                              │
│  • User Interaction Handling                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                                │
│                      (services/)                                 │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  DocumentIngestion│  │ DocumentProcessor│                    │
│  │  • File validation│  │ • Text splitting │                    │
│  │  • Document loading│  │ • Chunking      │                    │
│  └──────────────────┘  └──────────────────┘                    │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │VectorStoreManager│  │ RetrievalService │                    │
│  │ • Vector creation│  │ • Query processing│                    │
│  │ • Index updates  │  │ • Answer generation│                   │
│  └──────────────────┘  └──────────────────┘                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                                │
│                       (models/)                                  │
│  • HuggingFaceInferenceEmbeddings                               │
│  • Embedding Strategy Pattern                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                            │
│                      (config/)                                   │
│  • Environment Configuration                                     │
│  • API Key Management                                            │
│  • Model Initialization                                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│                                                                  │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐             │
│  │   Groq     │  │ HuggingFace  │  │   FAISS    │             │
│  │   (LLM)    │  │ (Embeddings) │  │  (Vector   │             │
│  │            │  │              │  │   Store)   │             │
│  └────────────┘  └──────────────┘  └────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
┌──────────┐
│  User    │
└────┬─────┘
     │ 1. Upload Document
     ▼
┌────────────────┐
│  app.py        │
│  (Streamlit)   │
└────┬───────────┘
     │ 2. Validate & Load
     ▼
┌──────────────────────┐
│ DocumentIngestion    │
│ • validate_file()    │
│ • load_document()    │
└────┬─────────────────┘
     │ 3. Split into Chunks
     ▼
┌──────────────────────┐
│ DocumentProcessor    │
│ • split_documents()  │
└────┬─────────────────┘
     │ 4. Generate Embeddings
     ▼
┌──────────────────────┐
│ HuggingFace API      │
│ • embed_documents()  │
└────┬─────────────────┘
     │ 5. Store Vectors
     ▼
┌──────────────────────┐
│ VectorStoreManager   │
│ • create_vector_store│
│ • FAISS Index        │
└────┬─────────────────┘
     │ 6. Save to Session
     ▼
┌──────────────────────┐
│ Session State        │
│ • st.session_state   │
└──────────────────────┘


QUERY PROCESSING FLOW:

┌──────────┐
│  User    │
└────┬─────┘
     │ 1. Ask Question
     ▼
┌──────────────────┐
│ RetrievalService │
└────┬─────────────┘
     │ 2. Embed Query
     ▼
┌──────────────────┐
│ HuggingFace API  │
└────┬─────────────┘
     │ 3. Similarity Search
     ▼
┌──────────────────┐
│ FAISS Vector DB  │
│ • Find Top 4     │
└────┬─────────────┘
     │ 4. Retrieve Chunks
     ▼
┌──────────────────┐
│ Document Chain   │
│ • Stuff Prompt   │
└────┬─────────────┘
     │ 5. Generate Answer
     ▼
┌──────────────────┐
│ Groq LLM         │
│ • ChatGroq       │
└────┬─────────────┘
     │ 6. Return Response
     ▼
┌──────────────────┐
│ User Interface   │
│ • Answer + Sources│
└──────────────────┘
```

---

## Component Architecture

### 1. Configuration Layer (`config/`)

**Responsibility**: Environment and configuration management

```
config/
├── __init__.py          # Package exports
└── settings.py          # Configuration constants and factories
```

**Key Components**:
- Environment variable loading (`.env`)
- API key management
- Model configuration constants
- Factory methods for LLM and embeddings
- Dependency checking

**Configuration Constants**:
```python
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_DOCUMENTS_TO_PROCESS = 50
EMBEDDING_BATCH_SIZE = 10
```

### 2. Models Layer (`models/`)

**Responsibility**: Embedding model implementations

```
models/
├── __init__.py          # Package exports
└── embeddings.py        # Embedding classes and factory
```

**Key Components**:
- `HuggingFaceInferenceEmbeddings`: Custom API-based embeddings
- `get_embeddings()`: Factory with automatic strategy selection
- Batch processing logic
- Error handling for API calls

**Embedding Dimensions**: 384-dimensional vectors (all-MiniLM-L6-v2)

### 3. Services Layer (`services/`)

**Responsibility**: Core business logic

```
services/
├── __init__.py          # Package exports
├── ingestion.py         # Document loading
├── processing.py        # Text processing
├── vectorstore.py       # Vector database management
└── retrieval.py         # Query processing
```

#### A. Document Ingestion Service

**Class**: `DocumentIngestion` (static methods)

**Responsibilities**:
- File validation (PDF, TXT)
- Document loading with appropriate loaders
- Temporary file management
- Error handling for failed loads
- Incremental file detection

**Key Methods**:
```python
validate_file(filename)              # Check file type
load_document(uploaded_file)         # Load single document
load_documents(uploaded_files)       # Batch loading
get_new_files(current, previous)     # Detect new uploads
```

#### B. Document Processing Service

**Class**: `DocumentProcessor`

**Responsibilities**:
- Text splitting into chunks
- Chunk size optimization
- Overlap management
- Document counting

**Splitting Strategy**: RecursiveCharacterTextSplitter
- Tries to split at paragraph boundaries first
- Falls back to sentence, then word boundaries
- Preserves semantic coherence

**Key Methods**:
```python
split_documents(documents, max_docs)  # Split into chunks
get_chunk_count(documents)            # Count resulting chunks
```

#### C. Vector Store Management Service

**Class**: `VectorStoreManager`

**Responsibilities**:
- FAISS vector store creation
- Incremental updates
- Vector store state management
- Chunk tracking

**Key Methods**:
```python
create_vector_store(documents)        # Initialize new store
update_vector_store(documents)        # Add new documents
get_vector_store()                    # Retrieve store
set_vector_store(store, count)        # Restore from session
```

**Optimization**: Merge strategy for incremental updates
- Creates temporary store for new documents
- Merges with existing store
- Avoids re-embedding all documents

#### D. Retrieval Service

**Class**: `RetrievalService`

**Responsibilities**:
- Retrieval chain creation
- Query processing
- Answer generation
- Performance tracking

**Key Methods**:
```python
create_retrieval_chain(vector_store)     # Build LangChain pipeline
process_query(query, vector_store)       # End-to-end query handling
get_relevant_documents(query, store, k)  # Retrieve without LLM
```

**Chain Architecture**:
```
Retriever → Document Chain → LLM → Response
```

### 4. Utilities Layer (`utils/`)

**Responsibility**: Reusable utilities

```
utils/
├── __init__.py          # Package exports
└── prompts.py           # Prompt templates
```

**Key Components**:
- `get_qa_prompt()`: Question-answering prompt template
- Prompt engineering best practices
- Template formatting

### 5. Presentation Layer (`app.py`)

**Responsibility**: UI and user interaction

**Key Functions**:
```python
initialize_components()              # Component initialization (cached)
initialize_session_state()           # Session state setup
create_vector_embeddings(files)      # Orchestrate embedding creation
process_query(query)                 # Orchestrate query processing
main()                               # Application entry point
```

**Session State Management**:
```python
st.session_state.vectors              # FAISS vector store
st.session_state.uploaded_file_names  # Processed files tracking
st.session_state.total_chunks         # Statistics
st.session_state.vector_store_manager # Service instance
```

---

## Data Flow

### Document Processing Pipeline

```
┌─────────────────────┐
│ 1. FILE UPLOAD      │
│ User uploads PDF    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 2. VALIDATION                           │
│ • Check file extension                  │
│ • Validate file type                    │
│ Result: (is_valid, file_extension)      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 3. TEMPORARY FILE CREATION              │
│ • Create temp file                      │
│ • Write uploaded content                │
│ • Get file path                         │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 4. DOCUMENT LOADING                     │
│ • PyPDFLoader for PDF                   │
│ • TextLoader for TXT                    │
│ Result: List[Document]                  │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 5. TEMP FILE CLEANUP                    │
│ • Delete temporary file                 │
│ • Free disk space                       │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 6. TEXT SPLITTING                       │
│ • RecursiveCharacterTextSplitter        │
│ • chunk_size=1000, overlap=200          │
│ Result: List[Document] (chunks)         │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 7. EMBEDDING GENERATION                 │
│ • Batch processing (size=10)            │
│ • HuggingFace API calls                 │
│ Result: List[Vector] (384-dim)          │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 8. VECTOR STORE CREATION/UPDATE         │
│ • FAISS.from_documents() OR             │
│ • FAISS.merge_from()                    │
│ Result: FAISS Index                     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 9. SESSION STATE UPDATE                 │
│ • Store vector store                    │
│ • Update file names list                │
│ • Update chunk count                    │
└─────────────────────────────────────────┘
```

### Query Processing Pipeline

```
┌─────────────────────┐
│ 1. USER QUERY       │
│ "What is X?"        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 2. QUERY VALIDATION                     │
│ • Check vector store exists             │
│ • Validate query is not empty           │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 3. QUERY EMBEDDING                      │
│ • Convert query to vector               │
│ • HuggingFace embed_query()             │
│ Result: Vector (384-dim)                │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 4. SIMILARITY SEARCH                    │
│ • FAISS.similarity_search()             │
│ • Find top k=4 similar chunks           │
│ • Cosine similarity calculation         │
│ Result: List[Document] (4 chunks)       │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 5. PROMPT CONSTRUCTION                  │
│ • Fill template with context            │
│ • Add user query                        │
│ Result: Formatted prompt string         │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 6. LLM GENERATION                       │
│ • Send prompt to Groq (ChatGroq)        │
│ • Temperature = 0 (deterministic)       │
│ • Wait for response                     │
│ Result: Answer string                   │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 7. RESPONSE FORMATTING                  │
│ • Package answer + context              │
│ • Add response time metadata            │
│ Result: Response dict                   │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ 8. UI DISPLAY                           │
│ • Show answer                           │
│ • Show source passages (expandable)     │
│ • Display performance metrics           │
└─────────────────────────────────────────┘
```

### State Management Flow

```
┌──────────────────────────────────────────┐
│         STREAMLIT EXECUTION MODEL        │
│                                          │
│  Every User Interaction Triggers         │
│  Complete Script Re-execution            │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│         SESSION STATE PERSISTENCE        │
│                                          │
│  st.session_state persists across reruns │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ vectors: FAISS Index               │ │
│  │ uploaded_file_names: List[str]     │ │
│  │ total_chunks: int                  │ │
│  │ vector_store_manager: Manager      │ │
│  │ show_embedding_success: bool       │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│        CACHED RESOURCES                  │
│                                          │
│  @st.cache_resource persists objects     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ llm: ChatGroq                      │ │
│  │ embeddings: HuggingFaceEmbeddings  │ │
│  │ prompt: ChatPromptTemplate         │ │
│  │ processor: DocumentProcessor       │ │
│  │ retrieval_service: RetrievalService│ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

---

## Technology Stack

### Core Framework
- **LangChain 0.3.7**: LLM application framework
  - Document loaders (PyPDF, TextLoader)
  - Text splitters (RecursiveCharacterTextSplitter)
  - Chains (Retrieval Chain, Document Chain)
  - Prompt templates

### LLM Provider
- **Groq**: Fast inference platform
  - Model: `meta-llama/llama-4-scout-17b-16e-instruct`
  - Temperature: 0 (deterministic)
  - Purpose: Answer generation

### Embeddings
- **HuggingFace**: Sentence transformers
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimensions: 384
  - Purpose: Semantic text representation

### Vector Database
- **FAISS (Facebook AI Similarity Search)**: In-memory vector store
  - Purpose: Fast similarity search
  - Index: Flat (brute force) for accuracy
  - Scalability: Suitable for up to millions of vectors

### Web Framework
- **Streamlit**: Python web application framework
  - Session state management
  - File uploaders
  - Interactive widgets
  - Caching mechanisms

### Document Processing
- **PyPDF**: PDF text extraction
- **Python-dotenv**: Environment variable management
- **tempfile**: Temporary file handling

### Python Version
- **Python 3.8+**: Required for LangChain and dependencies

---
