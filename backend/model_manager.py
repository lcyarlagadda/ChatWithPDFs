import fitz
import re
import os
import tiktoken
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from rank_bm25 import BM25Okapi
import time
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of ML models"""
    
    def __init__(self):
        self._embedding_model = None
        self._reranker = None
        self._mistral_model = None
        self._mistral_tokenizer = None
    
    def get_embedding_model(self):
        """Get or load the embedding model"""
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded")
        return self._embedding_model
    
    def get_reranker(self):
        """Get or load the reranker model"""
        if self._reranker is None:
            logger.info("Loading reranker model...")
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Reranker model loaded")
        return self._reranker
    
    def get_mistral_model(self):
        """Get or load the Mistral model"""
        if self._mistral_model is None or self._mistral_tokenizer is None:
            logger.info("Loading Mistral model...")
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_HUB_TOKEN environment variable.")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1", 
                token=hf_token
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            
            self._mistral_model = model
            self._mistral_tokenizer = tokenizer
            logger.info("Mistral model loaded")
        return self._mistral_model, self._mistral_tokenizer

