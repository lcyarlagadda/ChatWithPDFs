"""Response caching service"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """In-memory cache for RAG responses"""
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 7200):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, question: str, settings: Dict[str, Any]) -> str:
        """Generate cache key"""
        normalized = question.lower().strip()
        relevant = {
            'numChunks': settings.get('numChunks', 3),
            'chunkSize': settings.get('chunkSize', 500),
            'temperature': settings.get('temperature', 0.2),
            'maxTokens': settings.get('maxTokens', 512),
            'retrieverType': settings.get('retrieverType', 'Hybrid (Dense + Sparse)')
        }
        content = f"{normalized}|{json.dumps(relevant, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, question: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        key = self._generate_key(question, settings)
        
        if key not in self.cache:
            return None
        
        if time.time() - self.access_times[key] > self.ttl_seconds:
            self._remove_key(key)
            return None
        
        self.access_times[key] = time.time()
        logger.info(f"Cache hit: {question[:50]}...")
        return self.cache[key]
    
    def set(self, question: str, settings: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Cache response"""
        key = self._generate_key(question, settings)
        
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = response.copy()
        self.access_times[key] = time.time()
        logger.info(f"Cached: {question[:50]}...")
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict oldest item"""
        if not self.access_times:
            return
        oldest = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

