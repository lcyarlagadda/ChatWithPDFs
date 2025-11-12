"""RAG (Retrieval-Augmented Generation) service"""

import torch
import time
from typing import Dict, List
import logging

from utils.tokenizer import TokenCounter

logger = logging.getLogger(__name__)


class RAGService:
    """Service for question answering using RAG"""
    
    def __init__(self, retriever, model, tokenizer):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.token_counter = TokenCounter()
    
    def answer_question(self, question: str, k: int = 3) -> Dict:
        """Answer a question using RAG"""
        start_time = time.time()
        
        # Detect general questions
        is_general = any(phrase in question.lower() for phrase in [
            'what is', 'what are', 'overview', 'summary', 'about', 'content',
            'main topic', 'discuss', 'explain', 'describe'
        ])
        
        retrieval_k = min(k * 2, 8) if is_general else k
        retrieval_start = time.time()
        
        retrieved_chunks = self.retriever.retrieve(question, k=retrieval_k)
        retrieval_time = time.time() - retrieval_start
        
        # Optimize context window
        retrieved_chunks = self._optimize_context(retrieved_chunks, max_tokens=1800)
        
        # Build context with citations
        context_parts = []
        for chunk in retrieved_chunks:
            citation = chunk.get('citation', '')
            context_parts.append(f"{citation} {chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = self._build_prompt(context, question)
        generation_result = self._generate_response(prompt, max_length=512)
        
        total_latency = time.time() - start_time
        
        return {
            "question": question,
            "answer": generation_result["response"],
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "is_general_question": is_general,
            "retrieval_time": retrieval_time,
            "generation_time": generation_result["generation_time"],
            "input_tokens": generation_result["input_tokens"],
            "output_tokens": generation_result["output_tokens"],
            "total_latency": total_latency,
            "chunks_retrieved": len(retrieved_chunks),
            "context_length": len(context),
            "response_length": len(generation_result["response"])
        }
    
    def _build_prompt(self, context: str, question: str) -> str:
        """Build RAG prompt"""
        return f"""<s>[INST] Use ONLY the information provided in the context below. 
DO NOT use any external knowledge. 
Format the sentence properly with proper punctuation and capitalization and use simple english so that it is easy to understand.   
IMPORTANT: When citing information, use the exact citation format from the context (e.g., [document_name:page_number]).
If the question is generic, give overview or comparison as per the question.
If the question is specific and information is not in the context, state "I cannot find this information."
CONTEXT: {context}
QUESTION: {question}
Answer with proper citations using the format provided in the context.[/INST]"""
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> Dict:
        """Generate response using language model"""
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean response
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        if "CONTEXT:" in response:
            response = response.split("CONTEXT:")[0].strip()
        if "QUESTION:" in response:
            response = response.split("QUESTION:")[0].strip()
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        
        generation_time = time.time() - start_time
        
        return {
            "response": response,
            "input_tokens": self.token_counter.count(prompt),
            "output_tokens": self.token_counter.count(response),
            "generation_time": generation_time
        }
    
    def generate_streaming_response(self, prompt: str, max_length: int = 512):
        """Generate streaming response"""
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            for new_token in self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            ):
                token_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                
                if token_text in ["[INST]", "[/INST]", "CONTEXT:", "QUESTION:"]:
                    continue
                
                if new_token.item() == self.tokenizer.eos_token_id:
                    break
                
                yield {
                    "token": token_text,
                    "timestamp": time.time(),
                    "partial_response": True
                }
        
        generation_time = time.time() - start_time
        yield {
            "token": "",
            "timestamp": time.time(),
            "partial_response": False,
            "generation_time": generation_time,
            "input_tokens": self.token_counter.count(prompt),
            "stream_complete": True
        }
    
    def _optimize_context(self, chunks: List[Dict], max_tokens: int = 1800) -> List[Dict]:
        """Optimize context window to fit token limit"""
        prompt_overhead = 150
        available_tokens = max_tokens - prompt_overhead
        
        selected = []
        total_tokens = 0
        
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('final_score', x.get('combined_score', 0)),
            reverse=True
        )
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.get('token_count', self.token_counter.count(chunk['text']))
            
            if total_tokens + chunk_tokens <= available_tokens:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                if len(selected) == 0 and chunk_tokens <= available_tokens * 1.2:
                    selected.append(chunk)
                break
        
        return selected

