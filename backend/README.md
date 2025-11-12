# Backend Architecture Documentation

## Overview

The backend is a **modular FastAPI-based RAG (Retrieval-Augmented Generation) system** that processes PDF documents and answers questions using AI. It's organized into clear, reusable modules following separation of concerns.

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── api/                    # API layer
│   ├── __init__.py
│   ├── routes.py          # API endpoint handlers
│   └── schemas.py         # Pydantic request/response models
├── services/               # Business logic layer
│   ├── __init__.py
│   ├── pdf_service.py     # PDF processing service
│   ├── embedding_service.py  # Embedding generation
│   ├── retrieval_service.py   # Hybrid retrieval
│   ├── rag_service.py     # RAG pipeline
│   └── cache_service.py   # Response caching
├── core/                   # Core infrastructure
│   ├── __init__.py
│   └── models.py          # ML model management
└── utils/                  # Utility modules
    ├── __init__.py
    ├── text_processor.py  # Text cleaning & chunking
    └── tokenizer.py       # Token counting
```

## Module Descriptions

### API Layer (`api/`)

**Purpose**: Handles HTTP requests and responses

- **`routes.py`**: Defines all API endpoints
  - `POST /upload-pdfs`: Upload and process PDF files
  - `POST /ask-question`: Ask questions about documents
  - `POST /ask-question-stream`: Streaming question answering
  - `GET /health`: Health check
  - `GET /status`: System status
  - `POST /clear-cache`: Clear response cache
  - `GET /cache-stats`: Cache statistics

- **`schemas.py`**: Pydantic models for request/response validation
  - `QuestionRequest`: Question input schema
  - `QuestionResponse`: Answer output schema
  - `ProcessingStatus`: Processing status schema

### Services Layer (`services/`)

**Purpose**: Business logic and orchestration

#### `pdf_service.py` - PDF Processing
- **Responsibilities**:
  - Extract text from PDF files using PyMuPDF
  - Clean text (remove noise, headers, footers)
  - Chunk text into smaller pieces with metadata
  - Process multiple PDFs in parallel
  
- **Key Methods**:
  - `process_pdfs()`: Main entry point for PDF processing
  - `_extract_text_sync()`: Extract text from single PDF
  - `_chunk_text_sync()`: Split text into chunks with citations

#### `embedding_service.py` - Embedding Generation
- **Responsibilities**:
  - Generate vector embeddings for text chunks
  - Process embeddings in batches for efficiency
  - Use SentenceTransformer model (all-MiniLM-L6-v2)
  
- **Key Methods**:
  - `generate_embeddings()`: Generate embeddings for chunks
  - `generate_query_embedding()`: Generate embedding for query

#### `retrieval_service.py` - Hybrid Retrieval
- **Responsibilities**:
  - Combine dense (FAISS) and sparse (BM25) retrieval
  - Score and rank retrieved chunks
  - Apply reranking with CrossEncoder
  
- **Key Methods**:
  - `retrieve()`: Retrieve relevant chunks for query

#### `rag_service.py` - RAG Pipeline
- **Responsibilities**:
  - Orchestrate question answering pipeline
  - Retrieve relevant context
  - Generate answers using Mistral-7B
  - Optimize context window
  - Support streaming responses
  
- **Key Methods**:
  - `answer_question()`: Main Q&A method
  - `_generate_response()`: Generate answer from prompt
  - `generate_streaming_response()`: Stream tokens
  - `_optimize_context()`: Fit chunks in token limit

#### `cache_service.py` - Response Caching
- **Responsibilities**:
  - Cache question-answer pairs
  - TTL-based expiration
  - LRU eviction policy
  
- **Key Methods**:
  - `get()`: Retrieve cached response
  - `set()`: Cache response
  - `clear()`: Clear all cache

### Core Layer (`core/`)

**Purpose**: Infrastructure and model management

#### `models.py` - Model Manager
- **Responsibilities**:
  - Load and cache ML models
  - Lazy loading (load on demand)
  - Model initialization with quantization
  
- **Models Managed**:
  - Embedding model: `all-MiniLM-L6-v2` (384-dim)
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Language model: `mistralai/Mistral-7B-Instruct-v0.1` (4-bit quantized)

### Utils Layer (`utils/`)

**Purpose**: Reusable utility functions

#### `text_processor.py` - Text Processing
- **Functions**:
  - `clean_text()`: Fix PDF extraction issues
  - `filter_noise()`: Remove headers/footers
  - `split_into_sentences()`: Sentence splitting
  - `extract_metadata()`: Extract document/page info

#### `tokenizer.py` - Token Counting
- **Class**: `TokenCounter`
- **Methods**:
  - `count()`: Count tokens in text

## Request Flow

### PDF Upload Flow

```
1. Client uploads PDFs → POST /upload-pdfs
   ↓
2. API saves files temporarily
   ↓
3. PDFService.process_pdfs()
   - Extract text from all PDFs in parallel
   - Clean and filter noise
   - Chunk text with metadata
   ↓
4. EmbeddingService.generate_embeddings()
   - Generate 384-dim vectors for each chunk
   - Process in batches
   ↓
5. Create indexes
   - FAISS index for dense retrieval
   - BM25 index for sparse retrieval
   ↓
6. Initialize models (if needed)
   - Embedding model
   - Reranker
   - Mistral-7B language model
   ↓
7. Create HybridRetriever and RAGService
   ↓
8. Return success with chunk count
```

### Question Answering Flow

```
1. Client asks question → POST /ask-question
   ↓
2. Check ResponseCache
   - If cached and valid → return immediately
   ↓
3. RAGService.answer_question()
   a. Detect question type (general vs specific)
   b. HybridRetriever.retrieve()
      - Dense: Query embedding → FAISS search
      - Sparse: Query tokens → BM25 search
      - Combine scores (60% dense + 40% sparse)
      - Rerank with CrossEncoder
      - Select top K chunks
   ↓
   c. Optimize context window
      - Fit chunks within token limit (1800)
      - Prioritize highest-scoring chunks
   ↓
   d. Build prompt with context and citations
   ↓
   e. Generate answer with Mistral-7B
      - Temperature: 0.2 (low randomness)
      - Max tokens: 512
      - Clean response (remove artifacts)
   ↓
4. Extract citations from chunks
   ↓
5. Calculate metrics
   - Retrieval time
   - Generation time
   - Token counts
   - Total latency
   ↓
6. Cache response (if enabled)
   ↓
7. Return answer with citations and metrics
```

## Key Technologies

### ML Models
- **Mistral-7B-Instruct**: 7B parameter language model
  - Quantized to 4-bit for memory efficiency
  - Used for answer generation
  
- **all-MiniLM-L6-v2**: Sentence embedding model
  - 384-dimensional vectors
  - Used for semantic search
  
- **CrossEncoder**: Reranking model
  - Refines retrieval results
  - Improves relevance

### Libraries
- **FastAPI**: Modern async web framework
- **PyMuPDF (fitz)**: PDF text extraction
- **FAISS**: Vector similarity search
- **BM25**: Keyword-based search
- **SentenceTransformers**: Embedding generation
- **Transformers**: Hugging Face model loading
- **tiktoken**: Token counting

## Data Structures

### Chunk Structure
```python
{
    "id": 0,
    "text": "chunk text content...",
    "token_count": 450,
    "char_count": 1800,
    "document_name": "document.pdf",
    "page_number": 5,
    "citation": "[document.pdf:5]",
    "embedding": [0.123, 0.456, ...],  # 384-dim vector
    "dense_score": 0.85,
    "sparse_score": 0.72,
    "combined_score": 0.80,
    "rerank_score": 0.88,
    "final_score": 0.84,
    "rank": 1
}
```

## Configuration

### Environment Variables
- `HUGGINGFACE_HUB_TOKEN`: Required for loading Mistral model

### Default Settings
- Chunk size: 500 tokens
- Overlap: 50 tokens
- Number of chunks: 3
- Temperature: 0.2
- Max tokens: 512
- Batch size: 16-32 for embeddings
- Cache TTL: 7200 seconds (2 hours)
- Cache max size: 200 entries

## Performance Optimizations

1. **Parallel Processing**: Multiple PDFs processed simultaneously
2. **Batch Embeddings**: Process embeddings in batches
3. **Model Caching**: Models loaded once and reused
4. **Response Caching**: Cache question-answer pairs
5. **Async Operations**: Non-blocking I/O
6. **Context Optimization**: Dynamic chunk selection within token limits
7. **Quantization**: 4-bit model quantization for memory efficiency

## Error Handling

- **PDF Processing Errors**: Caught and logged, returns 500 error
- **Model Loading Errors**: Raises HTTPException with details
- **Missing PDFs**: Returns 400 error if no PDFs uploaded
- **Invalid Files**: Rejects non-PDF files
- **Timeout Handling**: Long operations have timeouts

## Security Considerations

- **CORS**: Configured for all origins (should be restricted in production)
- **File Validation**: Only PDF files accepted
- **Token Security**: Hugging Face token from environment
- **Input Sanitization**: Text cleaned before processing

## Running the Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Run server
python main.py
```

Server runs on `http://localhost:8000`

API documentation available at `http://localhost:8000/docs`

## Module Dependencies

```
main.py
  └── api/routes.py
       ├── services/pdf_service.py
       │    └── utils/text_processor.py
       │    └── utils/tokenizer.py
       ├── services/embedding_service.py
       ├── services/retrieval_service.py
       ├── services/rag_service.py
       │    └── utils/tokenizer.py
       ├── services/cache_service.py
       └── core/models.py
```

## Future Improvements

1. **Database Storage**: Persist chunks and embeddings
2. **WebSocket Support**: Real-time progress updates
3. **Multi-tenancy**: Support multiple users/sessions
4. **Advanced Reranking**: More sophisticated models
5. **Document Management**: Upload, delete, manage documents
6. **Export Features**: Export conversations, citations

