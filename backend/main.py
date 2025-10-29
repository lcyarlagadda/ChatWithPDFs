from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
import logging
from datetime import datetime
import json
import asyncio
import time

# Import our PDF processing modules
from pdf_processor import PDFProcessor
from rag_system import RAGSystem
from model_manager import ModelManager
from async_pdf_processor import AsyncPDFProcessor, ProcessingProgress
from async_embedding_generator import AsyncEmbeddingGenerator
from cache_manager import ResponseCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Chat API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the RAG system
rag_system: Optional[RAGSystem] = None
model_manager = ModelManager()
async_pdf_processor = AsyncPDFProcessor(max_workers=8)
async_embedding_generator = AsyncEmbeddingGenerator(batch_size=16)
response_cache = ResponseCache(max_size=200, ttl_seconds=7200)  # 2 hours TTL

class QuestionRequest(BaseModel):
    question: str
    settings: Dict[str, Any]
    use_streaming: bool = False
    use_cache: bool = True

class QuestionResponse(BaseModel):
    answer: str
    citations: List[str]
    metrics: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    from_cache: bool = False

class ProcessingStatus(BaseModel):
    status: str
    progress: Dict[str, Any]
    message: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    numChunks: int = Form(3),
    chunkSize: int = Form(500),
    temperature: float = Form(0.2),
    maxTokens: int = Form(512),
    modelType: str = Form("Mistral-7B"),
    retrieverType: str = Form("Hybrid (Dense + Sparse)"),
    enableNoiseFiltering: bool = Form(True),
    showChunks: bool = Form(False)
):
    """Upload and process PDF files with parallel processing"""
    global rag_system
    
    try:
        logger.info(f"Starting parallel processing of {len(files)} PDF files")
        start_time = time.time()
        
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Process PDFs in parallel
        result = await async_pdf_processor.process_pdfs_parallel(
            temp_files,
            progress_callback=None  # Can add WebSocket progress updates here
        )
        
        chunks = result["chunks"]
        
        # Generate embeddings in parallel
        logger.info("Generating embeddings in parallel...")
        chunks_with_embeddings = await async_embedding_generator.generate_embeddings_parallel(chunks)
        
        # Create FAISS index
        import numpy as np
        import faiss
        from rank_bm25 import BM25Okapi
        
        embeddings_array = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_array.astype('float32'))
        
        # Create BM25 index
        texts = [chunk["text"] for chunk in chunks_with_embeddings]
        tokenized_texts = [text.split() for text in texts]
        bm25_index = BM25Okapi(tokenized_texts)
        
        # Initialize models
        embedding_model = model_manager.get_embedding_model()
        reranker = model_manager.get_reranker()
        
        # Create retriever
        from retriever import SimpleRetriever
        retriever = SimpleRetriever(embedding_model, faiss_index, bm25_index, {"chunks": chunks_with_embeddings}, reranker)
        
        # Load language model
        mistral_model, mistral_tokenizer = model_manager.get_mistral_model()
        
        # Create RAG system
        rag_system = RAGSystem(retriever, mistral_model, mistral_tokenizer)
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {len(chunks)} chunks in {processing_time:.2f}s")
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(files)} PDF(s) with {len(chunks)} chunks",
            "chunks_count": len(chunks),
            "files_processed": len(files),
            "processing_time": processing_time,
            "pages_processed": result.get("pages_processed", 0)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded PDFs with caching support"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet. Please upload PDFs first.")
    
    try:
        logger.info(f"Processing question: {request.question[:50]}...")
        
        # Check cache first
        cached_response = None
        if request.use_cache:
            cached_response = response_cache.get(request.question, request.settings)
        
        if cached_response:
            logger.info("Returning cached response")
            return QuestionResponse(
                answer=cached_response['answer'],
                citations=cached_response['citations'],
                metrics=cached_response['metrics'],
                retrieved_chunks=cached_response.get('retrieved_chunks', []),
                from_cache=True
            )
        
        # Update settings if provided
        settings = request.settings
        k = settings.get('numChunks', 3)
        
        # Get answer from RAG system
        result = rag_system.answer_question(request.question, k=k)
        
        # Extract citations
        citations = []
        if 'retrieved_chunks' in result:
            seen_citations = set()
            for chunk in result['retrieved_chunks']:
                citation = chunk.get('citation', '')
                if citation and citation not in seen_citations:
                    citations.append(citation)
                    seen_citations.add(citation)
        
        # Prepare metrics
        metrics = {
            "retrieval_time": result.get('retrieval_time', 0),
            "generation_time": result.get('generation_time', 0),
            "input_tokens": result.get('input_tokens', 0),
            "output_tokens": result.get('output_tokens', 0),
            "total_latency": result.get('total_latency', 0),
            "chunks_retrieved": len(result.get('retrieved_chunks', [])),
            "context_length": len(result.get('context', '')),
            "response_length": len(result.get('answer', ''))
        }
        
        response_data = {
            "answer": result['answer'],
            "citations": citations,
            "metrics": metrics,
            "retrieved_chunks": result.get('retrieved_chunks', [])
        }
        
        # Cache the response
        if request.use_cache:
            response_cache.set(request.question, request.settings, response_data)
        
        return QuestionResponse(**response_data, from_cache=False)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask-question-stream")
async def ask_question_stream(request: QuestionRequest):
    """Ask a question with streaming response"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet. Please upload PDFs first.")
    
    async def generate_stream():
        try:
            logger.info(f"Processing streaming question: {request.question[:50]}...")
            
            # Update settings if provided
            settings = request.settings
            k = settings.get('numChunks', 3)
            
            # Get retrieved chunks first
            retrieved_chunks = rag_system.retriever.retrieve(request.question, k=k)
            
            # Create context with citations
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                citation = chunk.get('citation', f"[{i+1}]")
                context_parts.append(f"{citation} {chunk['text']}")
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt
            rag_prompt = f"""<s>[INST] Use ONLY the information provided in the context below. 
            DO NOT use any external knowledge. 
            Format the sentence properly with proper punctuation and capitalization and use simple english so that it is easy to understand.   
            IMPORTANT: When citing information, use the exact citation format from the context (e.g., [document_name:page_number]).
            If the question is generic, give overview or comparison as per the question.
            If the question is specific and information is not in the context, state "I cannot find this information."
            CONTEXT: {context}
            QUESTION: {request.question}
            Answer with proper citations using the format provided in the context.[/INST]"""
            
            # Stream response
            full_response = ""
            async for token_data in rag_system.generate_streaming_response(rag_prompt, max_length=512):
                if token_data.get('partial_response', False):
                    token = token_data.get('token', '')
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'partial': True})}\n\n"
                elif token_data.get('stream_complete', False):
                    # Send final metrics
                    metrics = {
                        "generation_time": token_data.get('generation_time', 0),
                        "input_tokens": token_data.get('input_tokens', 0),
                        "output_tokens": len(rag_system.tokenizer.encode(full_response)),
                        "response_length": len(full_response)
                    }
                    yield f"data: {json.dumps({'complete': True, 'metrics': metrics, 'full_response': full_response})}\n\n"
                    break
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.get("/status")
async def get_status():
    """Get current system status"""
    global rag_system
    
    return {
        "rag_system_loaded": rag_system is not None,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "embedding_model": model_manager._embedding_model is not None,
            "reranker": model_manager._reranker is not None,
            "mistral_model": model_manager._mistral_model is not None
        },
        "cache_stats": response_cache.get_stats(),
        "async_processor_ready": async_pdf_processor is not None,
        "async_embedding_ready": async_embedding_generator is not None
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear response cache"""
    response_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    return response_cache.get_stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

