"""Hybrid retrieval service"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval"""
    
    def __init__(
        self,
        embedding_model,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunks: List[Dict],
        reranker: Optional[object] = None
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.reranker = reranker
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks using hybrid approach"""
        # Dense retrieval
        query_embedding = self.embedding_model.encode([query])[0]
        distances, dense_indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k * 2
        )
        
        # Sparse retrieval
        bm25_scores = self.bm25_index.get_scores(query.split())
        sparse_indices = np.argsort(bm25_scores)[::-1][:k * 2]
        
        # Combine results
        all_indices = list(set(dense_indices[0].tolist() + sparse_indices.tolist()))
        
        chunks_with_scores = []
        for idx in all_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                
                # Calculate scores
                dense_score = 0
                if idx in dense_indices[0]:
                    dense_idx = np.where(dense_indices[0] == idx)[0]
                    if len(dense_idx) > 0:
                        dense_score = 1 / (1 + distances[0][dense_idx[0]])
                
                sparse_score = bm25_scores[idx] / (np.max(bm25_scores) + 1e-8)
                combined_score = 0.6 * dense_score + 0.4 * sparse_score
                
                chunk["dense_score"] = round(dense_score, 3)
                chunk["sparse_score"] = round(sparse_score, 3)
                chunk["combined_score"] = round(combined_score, 3)
                chunks_with_scores.append(chunk)
        
        # Sort by combined score
        chunks_with_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Rerank if available
        if self.reranker and len(chunks_with_scores) > k:
            query_chunk_pairs = [(query, chunk["text"]) for chunk in chunks_with_scores[:k * 2]]
            rerank_scores = self.reranker.predict(query_chunk_pairs)
            
            for i, chunk in enumerate(chunks_with_scores[:k * 2]):
                chunk["rerank_score"] = round(rerank_scores[i], 3)
                chunk["final_score"] = round(0.7 * chunk["combined_score"] + 0.3 * rerank_scores[i], 3)
        else:
            for chunk in chunks_with_scores:
                chunk["final_score"] = chunk["combined_score"]
        
        chunks_with_scores.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add ranking
        for i, chunk in enumerate(chunks_with_scores[:k]):
            chunk["rank"] = i + 1
        
        return chunks_with_scores[:k]

