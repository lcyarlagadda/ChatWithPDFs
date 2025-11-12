"""Token counting utilities"""

import tiktoken
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """Handles token counting for text"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize tokenizer with specified encoding"""
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load {encoding_name}, using gpt-3.5-turbo encoding")
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using word approximation")
            return len(text.split()) * 1.3  # Approximate fallback

