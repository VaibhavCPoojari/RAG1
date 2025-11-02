# RAG Pipeline - Modular Architecture

A modular Retrieval-Augmented Generation (RAG) pipeline for document question-answering using LangChain, Streamlit, and various AI models.

## Project Structure

```
rag-pipeline-vaibhavcpoojari/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── .env                           # Environment variables (not in repo)
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration and environment setup
├── models/
│   ├── __init__.py
│   └── embeddings.py              # Custom embedding models
├── services/
│   ├── __init__.py
│   ├── ingestion.py               # Document loading and file handling
│   ├── processing.py              # Text splitting and preprocessing
│   ├── vectorstore.py             # Vector store management
│   └── retrieval.py               # Query processing and retrieval
├── utils/
│   ├── __init__.py
│   └── prompts.py                 # Prompt templates
└── researchpapers/                # Document storage directory
```

## Module Overview

### 1. Configuration (`config/`)

**`settings.py`**
- Manages environment variables and API keys
- Initializes LLM and embedding models
- Contains all configuration constants
- Provides utility functions for model initialization

### 2. Models (`models/`)

**`embeddings.py`**
- `HuggingFaceInferenceEmbeddings`: Custom embeddings using HuggingFace API
- `get_embeddings()`: Factory function for embeddings initialization
- Supports both API-based and local embeddings

### 3. Services (`services/`)

**`ingestion.py`** - Document Ingestion Service
- File validation and type checking
- Document loading (PDF, TXT)
- Temporary file management
- Batch document processing
- New file detection

**`processing.py`** - Document Processing Service
- Text splitting using RecursiveCharacterTextSplitter
- Configurable chunk size and overlap
- Document preprocessing
- Chunk counting utilities

**`vectorstore.py`** - Vector Store Management
- FAISS vector store creation
- Vector store updates and merging
- Chunk tracking
- Vector store persistence

**`retrieval.py`** - Retrieval Service
- Retrieval chain creation
- Query processing
- Response generation with timing
- Document similarity search

### 4. Utilities (`utils/`)

**`prompts.py`**
- Prompt template definitions
- Prompt formatting utilities
- Reusable prompt components

### 5. Main Application (`app.py`)

- Streamlit UI orchestration
- Session state management
- Component initialization
- User interaction handling
- Response display

## Key Features

### Modular Design Benefits

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Reusability**: Components can be used independently or in other projects
3. **Testability**: Easy to write unit tests for individual modules
4. **Maintainability**: Changes are isolated to specific modules
5. **Scalability**: Easy to extend with new features

### Functionality

- Multiple file format support (PDF, TXT)
- Incremental document processing
- Vector store persistence
- Document similarity search
- Response time tracking
- Source passage viewing

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-pipeline-vaibhavcpoojari
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Documents**: Upload PDF or TXT files using the file uploader
2. **Create Embeddings**: Click "Create Document Embeddings" to process documents
3. **Ask Questions**: Enter your query in the text input
4. **View Results**: See the answer and source passages

## Configuration Options

Edit `config/settings.py` to customize:

- `LLM_MODEL_NAME`: Language model to use
- `LLM_TEMPERATURE`: Temperature for response generation
- `EMBEDDING_MODEL_NAME`: Embedding model
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_DOCUMENTS_TO_PROCESS`: Maximum documents to process (default: 50)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: 10)

## Extending the Application

### Adding New Document Types

1. Add loader in `services/ingestion.py`
2. Update `SUPPORTED_EXTENSIONS` list
3. Add loading logic in `load_document()` method

### Adding New Embedding Models

1. Create new embedding class in `models/embeddings.py`
2. Update `get_embeddings()` factory function
3. Add configuration in `config/settings.py`

### Custom Prompts

1. Add new prompt template in `utils/prompts.py`
2. Import and use in `app.py` or service modules

## Dependencies

- **LangChain**: Framework for LLM applications
- **Streamlit**: Web interface
- **FAISS**: Vector store
- **HuggingFace**: Embeddings and models
- **Groq**: LLM provider
- **PyPDF**: PDF processing
- **python-dotenv**: Environment management

## Architecture Diagram

```
User Interface (Streamlit)
         |
         v
    app.py (Orchestrator)
         |
         +-- config/settings.py (Configuration)
         |
         +-- models/embeddings.py (Embeddings)
         |
         +-- services/
         |    |
         |    +-- ingestion.py (Load docs)
         |    |
         |    +-- processing.py (Split text)
         |    |
         |    +-- vectorstore.py (Create vectors)
         |    |
         |    +-- retrieval.py (Query & Response)
         |
         +-- utils/prompts.py (Prompt templates)
```

 
