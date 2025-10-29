import hashlib
import json
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple in-memory cache for RAG responses"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, question: str, settings: Dict[str, Any]) -> str:
        """Generate cache key from question and settings"""
        # Normalize question (lowercase, strip whitespace)
        normalized_question = question.lower().strip()
        
        # Create settings hash (only include relevant settings)
        relevant_settings = {
            'numChunks': settings.get('numChunks', 3),
            'chunkSize': settings.get('chunkSize', 500),
            'temperature': settings.get('temperature', 0.2),
            'maxTokens': settings.get('maxTokens', 512),
            'retrieverType': settings.get('retrieverType', 'Hybrid (Dense + Sparse)')
        }
        
        # Create hash
        content = f"{normalized_question}|{json.dumps(relevant_settings, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, question: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        key = self._generate_key(question, settings)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.access_times[key] > self.ttl_seconds:
            self._remove_key(key)
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        
        logger.info(f"Cache hit for question: {question[:50]}...")
        return self.cache[key]
    
    def set(self, question: str, settings: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Cache response"""
        key = self._generate_key(question, settings)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Store response
        self.cache[key] = response.copy()
        self.access_times[key] = time.time()
        
        logger.info(f"Cached response for question: {question[:50]}...")
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_oldest(self) -> None:
        """Remove oldest accessed item"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
    
    def clear(self) -> None:
        """Clear all cached responses"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": getattr(self, '_hits', 0) / max(getattr(self, '_total_requests', 1), 1)
        }
    
    def invalidate_similar(self, question: str, similarity_threshold: float = 0.8) -> int:
        """Invalidate cache entries similar to given question (simple implementation)"""
        normalized_question = question.lower().strip()
        removed_count = 0
        
        keys_to_remove = []
        for key, cached_response in self.cache.items():
            cached_question = cached_response.get('question', '').lower().strip()
            
            # Simple similarity check (can be improved with embeddings)
            if self._calculate_similarity(normalized_question, cached_question) > similarity_threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_key(key)
            removed_count += 1
        
        logger.info(f"Invalidated {removed_count} similar cache entries")
        return removed_count
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for text comparison"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
