"""API route handlers"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
import os
import tempfile
import json
import time
import logging

from .schemas import QuestionRequest, QuestionResponse
from ..services.pdf_service import PDFService
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import HybridRetriever
from ..services.rag_service import RAGService
from ..services.cache_service import ResponseCache
from ..services.llama_index_service import LlamaIndexService
from ..core.models import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state
rag_service: Optional[RAGService] = None
llama_index_service: Optional[LlamaIndexService] = None
pdf_service = PDFService(max_workers=8)
embedding_service = EmbeddingService(batch_size=16)
model_manager = ModelManager()
response_cache = ResponseCache(max_size=200, ttl_seconds=7200)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.post("/upload-pdfs")
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
    """Upload and process PDF files"""
    global rag_service, llama_index_service
    
    try:
        logger.info(f"Processing {len(files)} PDF files")
        start_time = time.time()
        
        # Save uploaded files
        temp_files = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Process PDFs
        result = await pdf_service.process_pdfs(temp_files, chunk_size=chunkSize)
        chunks = result["chunks"]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunks_with_embeddings = await embedding_service.generate_embeddings(chunks)
        
        # Create indexes
        import numpy as np
        import faiss
        from rank_bm25 import BM25Okapi
        
        embeddings_array = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_array.astype('float32'))
        
        texts = [chunk["text"] for chunk in chunks_with_embeddings]
        tokenized_texts = [text.split() for text in texts]
        bm25_index = BM25Okapi(tokenized_texts)
        
        # Initialize models
        embedding_model = model_manager.get_embedding_model()
        reranker = model_manager.get_reranker()
        
        # Create retriever
        retriever = HybridRetriever(
            embedding_model,
            faiss_index,
            bm25_index,
            chunks_with_embeddings,
            reranker
        )
        
        # Load language model
        mistral_model, mistral_tokenizer = model_manager.get_mistral_model()
        
        # Create RAG service
        rag_service = RAGService(retriever, mistral_model, mistral_tokenizer)

        # Create LlamaIndex retriever
        try:
            llama_index_service = LlamaIndexService()
            llama_index_service.build_index(chunks_with_embeddings)
            logger.info("LlamaIndex retriever initialized")
        except Exception as e:
            llama_index_service = None
            logger.warning(f"LlamaIndex initialization failed: {e}")
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(chunks)} chunks in {processing_time:.2f}s")
        
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


@router.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded PDFs"""
    global rag_service, llama_index_service
    
    if rag_service is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet. Please upload PDFs first.")
    
    try:
        logger.info(f"Processing question: {request.question[:50]}...")
        
        # Check cache
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
        
        retriever_type = request.settings.get('retrieverType', 'Hybrid (Dense + Sparse)')
        retriever_override = None
        if retriever_type == 'LlamaIndex':
            if llama_index_service is None or not llama_index_service.is_ready():
                raise HTTPException(status_code=400, detail="LlamaIndex retriever not ready. Please upload PDFs again.")
            retriever_override = llama_index_service

        # Get answer
        k = request.settings.get('numChunks', 3)
        result = rag_service.answer_question(
            request.question,
            k=k,
            retriever=retriever_override
        )
        
        # Extract citations
        citations = []
        seen = set()
        for chunk in result['retrieved_chunks']:
            citation = chunk.get('citation', '')
            if citation and citation not in seen:
                citations.append(citation)
                seen.add(citation)
        
        # Prepare metrics
        metrics = {
            "retrieval_time": result.get('retrieval_time', 0),
            "generation_time": result.get('generation_time', 0),
            "input_tokens": result.get('input_tokens', 0),
            "output_tokens": result.get('output_tokens', 0),
            "total_latency": result.get('total_latency', 0),
            "chunks_retrieved": len(result.get('retrieved_chunks', [])),
            "context_length": len(result.get('context', '')),
            "response_length": len(result.get('answer', '')),
            "retriever_type": retriever_type
        }
        
        response_data = {
            "answer": result['answer'],
            "citations": citations,
            "metrics": metrics,
            "retrieved_chunks": result.get('retrieved_chunks', [])
        }
        
        # Cache response
        if request.use_cache:
            response_cache.set(request.question, request.settings, response_data)
        
        return QuestionResponse(**response_data, from_cache=False)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/ask-question-stream")
async def ask_question_stream(request: QuestionRequest):
    """Ask question with streaming response"""
    global rag_service, llama_index_service
    
    if rag_service is None:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet. Please upload PDFs first.")
    
    async def generate_stream():
        try:
            logger.info(f"Processing streaming question: {request.question[:50]}...")
            
            retriever_type = request.settings.get('retrieverType', 'Hybrid (Dense + Sparse)')
            active_retriever = rag_service.retriever
            if retriever_type == 'LlamaIndex':
                if llama_index_service is None or not llama_index_service.is_ready():
                    raise HTTPException(status_code=400, detail="LlamaIndex retriever not ready. Please upload PDFs again.")
                active_retriever = llama_index_service

            k = request.settings.get('numChunks', 3)
            retrieved_chunks = active_retriever.retrieve(request.question, k=k)
            
            context_parts = []
            for chunk in retrieved_chunks:
                citation = chunk.get('citation', '')
                context_parts.append(f"{citation} {chunk['text']}")
            context = "\n\n".join(context_parts)
            
            prompt = rag_service._build_prompt(context, request.question)
            
            full_response = ""
            for token_data in rag_service.generate_streaming_response(prompt, max_length=512):
                if token_data.get('partial_response', False):
                    token = token_data.get('token', '')
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'partial': True})}\n\n"
                elif token_data.get('stream_complete', False):
                    from ..utils.tokenizer import TokenCounter
                    token_counter = TokenCounter()
                    metrics = {
                        "generation_time": token_data.get('generation_time', 0),
                        "input_tokens": token_data.get('input_tokens', 0),
                        "output_tokens": token_counter.count(full_response),
                        "response_length": len(full_response),
                        "retriever_type": retriever_type
                    }
                    yield f"data: {json.dumps({'complete': True, 'metrics': metrics, 'full_response': full_response})}\n\n"
                    break
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.get("/status")
async def get_status():
    """Get system status"""
    global rag_service
    
    return {
        "rag_service_loaded": rag_service is not None,
        "llama_index_ready": llama_index_service is not None and llama_index_service.is_ready(),
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "models_loaded": {
            "embedding_model": model_manager._embedding_model is not None,
            "reranker": model_manager._reranker is not None,
            "mistral_model": model_manager._mistral_model is not None
        },
        "cache_stats": response_cache.get_stats()
    }


@router.post("/clear-cache")
async def clear_cache():
    """Clear response cache"""
    response_cache.clear()
    return {"message": "Cache cleared successfully"}


@router.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    return response_cache.get_stats()

