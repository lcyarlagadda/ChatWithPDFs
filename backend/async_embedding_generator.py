import asyncio
import numpy as np
from typing import List, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import time

logger = logging.getLogger(__name__)

class AsyncEmbeddingGenerator:
    """Generate embeddings asynchronously with batching and parallel processing"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, max_workers: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def initialize_model(self):
        """Initialize the embedding model asynchronously"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )
            logger.info("Embedding model loaded successfully")
    
    def _load_model_sync(self):
        """Load model synchronously"""
        return SentenceTransformer(self.model_name)
    
    async def generate_embeddings_batch(self, texts: List[str], progress_callback=None) -> np.ndarray:
        """Generate embeddings for texts in batches"""
        if self.model is None:
            await self.initialize_model()
        
        total_texts = len(texts)
        all_embeddings = []
        
        logger.info(f"Generating embeddings for {total_texts} texts in batches of {self.batch_size}")
        
        # Process in batches
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Update progress
            if progress_callback:
                progress = {
                    "processed": min(i + self.batch_size, total_texts),
                    "total": total_texts,
                    "percentage": (min(i + self.batch_size, total_texts) / total_texts) * 100
                }
                await progress_callback(progress)
        
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated {len(final_embeddings)} embeddings")
        
        return final_embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_batch_embeddings_sync,
            texts
        )
    
    def _generate_batch_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings synchronously for a batch"""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 384))  # 384 is the dimension for all-MiniLM-L6-v2
    
    async def generate_embeddings_parallel(self, chunks: List[Dict], progress_callback=None) -> List[Dict]:
        """Generate embeddings for chunks in parallel with progress tracking"""
        start_time = time.time()
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings_batch(texts, progress_callback)
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()
            enhanced_chunk["embedding"] = embeddings[i].tolist()
            enhanced_chunks.append(enhanced_chunk)
        
        processing_time = time.time() - start_time
        logger.info(f"Enhanced {len(chunks)} chunks with embeddings in {processing_time:.2f}s")
        
        return enhanced_chunks
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if self.model is None:
            await self.initialize_model()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_query_embedding_sync,
            query
        )
    
    def _generate_query_embedding_sync(self, query: str) -> np.ndarray:
        """Generate query embedding synchronously"""
        try:
            return self.model.encode([query])[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return np.zeros(384)  # Fallback
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
