"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel
from typing import List, Dict, Any


class QuestionRequest(BaseModel):
    """Request schema for asking questions"""
    question: str
    settings: Dict[str, Any]
    use_streaming: bool = False
    use_cache: bool = True


class QuestionResponse(BaseModel):
    """Response schema for questions"""
    answer: str
    citations: List[str]
    metrics: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    from_cache: bool = False


class ProcessingStatus(BaseModel):
    """Status schema for processing"""
    status: str
    progress: Dict[str, Any]
    message: str

