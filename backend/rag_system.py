import torch
import time
import re
from typing import Dict, List
import logging
from pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG system for question answering"""
    
    def __init__(self, retriever, model, tokenizer):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.pdf_processor = PDFProcessor()
    
    def answer_question(self, question: str, k: int = 3) -> Dict:
        """Answer a question using RAG"""
        query_start_time = time.time()
        
        # For general questions, retrieve more chunks to get better overview
        is_general_question = any(phrase in question.lower() for phrase in [
            'what is', 'what are', 'overview', 'summary', 'about', 'content', 
            'main topic', 'discuss', 'explain', 'describe'
        ])
        
        # Use more chunks for general questions to get better context
        retrieval_k = min(k * 2, 8) if is_general_question else k
        retrieval_start = time.time()
        
        # Retrieve using the original query
        retrieved_chunks = self.retriever.retrieve(question, k=retrieval_k)
        retrieval_time = time.time() - retrieval_start
        
        # Optimize context window to fit token limit
        retrieved_chunks = self.optimize_context_window(retrieved_chunks, question, max_tokens=1800)
        
        # Create context with citations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            # Use metadata citation if available, otherwise use index
            citation = chunk.get('citation', f"[{i+1}]")
            context_parts.append(f"{citation} {chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with more flexible instructions
        rag_prompt = f"""<s>[INST] Use ONLY the information provided in the context below. 
        DO NOT use any external knowledge. 
        Format the sentence properly with proper punctuation and capitalization and use simple english so that it is easy to understand.   
        IMPORTANT: When citing information, use the exact citation format from the context (e.g., [document_name:page_number]).
        If the question is generic, give overview or comparison as per the question.
        If the question is specific and information is not in the context, state "I cannot find this information."
        CONTEXT: {context}
        QUESTION: {question}
        Answer with proper citations using the format provided in the context.[/INST]"""
        
        # Generate response
        generation_result = self.generate_response(rag_prompt, max_length=512)
        response = generation_result.get("response", "")
        response_metrics = generation_result
        
        # Calculate and log metrics
        total_latency = time.time() - query_start_time
        
        metrics = {
            "retrieval_time": retrieval_time,
            "generation_time": response_metrics.get("generation_time", 0),
            "input_tokens": response_metrics.get("input_tokens", 0),
            "output_tokens": response_metrics.get("output_tokens", 0),
            "total_latency": total_latency,
            "chunks_retrieved": len(retrieved_chunks),
            "context_length": len(context),
            "response_length": len(response)
        }
        
        return {
            "question": question, 
            "answer": response, 
            "retrieved_chunks": retrieved_chunks, 
            "context": context,
            "is_general_question": is_general_question,
            "retrieval_count": len(retrieved_chunks),
            **metrics
        }
    
    def generate_response(self, prompt: str, max_length: int = 512) -> Dict:
        """Generate response using the language model"""
        generation_start = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to same device as model
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
        
        # Clean up the response to show only the actual answer
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        # Remove any remaining prompt artifacts
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        # Remove any remaining context or question references
        if "CONTEXT:" in response:
            response = response.split("CONTEXT:")[0].strip()
        if "QUESTION:" in response:
            response = response.split("QUESTION:")[0].strip()
        
        # Clean up any remaining artifacts
        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
        
        # Log generation metrics
        generation_time = time.time() - generation_start
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(response))
        
        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_time": generation_time
        }
    
    def generate_streaming_response(self, prompt: str, max_length: int = 512):
        """Generate streaming response using the language model"""
        generation_start = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with streaming
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
                # Decode only the new token
                token_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                
                # Clean up artifacts as they come
                if token_text in ["[INST]", "[/INST]", "CONTEXT:", "QUESTION:"]:
                    continue
                
                # Check for end of generation
                if new_token.item() == self.tokenizer.eos_token_id:
                    break
                
                yield {
                    "token": token_text,
                    "timestamp": time.time(),
                    "partial_response": True
                }
        
        # Final metrics
        generation_time = time.time() - generation_start
        input_tokens = len(self.tokenizer.encode(prompt))
        
        yield {
            "token": "",
            "timestamp": time.time(),
            "partial_response": False,
            "generation_time": generation_time,
            "input_tokens": input_tokens,
            "stream_complete": True
        }
    
    def optimize_context_window(self, chunks: List[Dict], question: str, max_tokens: int = 1800) -> List[Dict]:
        """Dynamically select chunks within token limit"""
        # Calculate tokens for prompt structure
        prompt_overhead = 150  # Approximate tokens for prompt template
        available_tokens = max_tokens - prompt_overhead
        
        selected_chunks = []
        total_tokens = 0
        
        # Sort chunks by score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.get('final_score', x.get('combined_score', 0)), reverse=True)
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.get('token_count', self.pdf_processor.count_tokens(chunk['text']))
            
            # Check if adding this chunk would exceed limit
            if total_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to include at least the top chunk even if slightly over
                if len(selected_chunks) == 0 and chunk_tokens <= available_tokens * 1.2:
                    selected_chunks.append(chunk)
                    total_tokens += chunk_tokens
                break
        
        return selected_chunks

