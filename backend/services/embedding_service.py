"""Embedding generation service"""

import asyncio
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, max_workers: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def initialize(self):
        """Initialize embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                lambda: SentenceTransformer(self.model_name)
            )
            logger.info("Embedding model loaded")
    
    async def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks"""
        if self.model is None:
            await self.initialize()
        
        start_time = time.time()
        texts = [chunk["text"] for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = await self._generate_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced = chunk.copy()
            enhanced["embedding"] = embeddings[i].tolist()
            enhanced_chunks.append(enhanced)
        
        logger.info(f"Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s")
        return enhanced_chunks
    
    async def _generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(texts, show_progress_bar=False)
        )
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if self.model is None:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode([query])[0]
        )

