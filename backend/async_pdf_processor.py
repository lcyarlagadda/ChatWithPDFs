import asyncio
import aiofiles
import fitz
import re
import os
import tiktoken
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class ProcessingProgress:
    """Track processing progress for large documents"""
    total_pages: int
    processed_pages: int
    total_chunks: int
    processed_chunks: int
    current_file: str
    status: str  # 'processing', 'chunking', 'embedding', 'indexing', 'complete'
    start_time: float
    estimated_completion: Optional[float] = None

class AsyncPDFProcessor:
    """Async PDF processor with parallel processing capabilities"""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 500, overlap: int = 50):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Thread pool for I/O operations
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        # Process pool for CPU-intensive operations
        self.cpu_executor = ProcessPoolExecutor(max_workers=max(1, self.max_workers // 2))
        
        logger.info(f"Initialized AsyncPDFProcessor with {self.max_workers} workers")
    
    async def process_pdfs_parallel(self, file_paths: List[str], progress_callback=None) -> Dict:
        """Process multiple PDFs in parallel"""
        start_time = time.time()
        
        # Create progress tracker
        progress = ProcessingProgress(
            total_pages=0,
            processed_pages=0,
            total_chunks=0,
            processed_chunks=0,
            current_file="",
            status="processing",
            start_time=start_time
        )
        
        try:
            # Step 1: Extract text from all PDFs in parallel
            logger.info(f"Starting parallel processing of {len(file_paths)} PDFs")
            progress.status = "processing"
            
            # Get page counts first for progress tracking
            page_counts = await self._get_page_counts_parallel(file_paths)
            progress.total_pages = sum(page_counts)
            
            # Extract text from all PDFs in parallel
            extraction_tasks = [
                self._extract_text_async(file_path, progress_callback, progress)
                for file_path in file_paths
            ]
            
            extracted_texts = await asyncio.gather(*extraction_tasks)
            
            # Step 2: Combine and chunk text
            logger.info("Combining extracted texts and creating chunks")
            progress.status = "chunking"
            
            combined_text = self._combine_texts(extracted_texts, file_paths)
            chunks = await self._chunk_text_async(combined_text, progress_callback, progress)
            
            progress.total_chunks = len(chunks)
            progress.processed_chunks = len(chunks)
            progress.status = "complete"
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(file_paths)} PDFs in {processing_time:.2f}s")
            
            return {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "files_processed": len(file_paths),
                "processing_time": processing_time,
                "pages_processed": progress.total_pages
            }
            
        except Exception as e:
            logger.error(f"Error in parallel PDF processing: {str(e)}")
            raise
        finally:
            # Cleanup
            self.io_executor.shutdown(wait=False)
            self.cpu_executor.shutdown(wait=False)
    
    async def _get_page_counts_parallel(self, file_paths: List[str]) -> List[int]:
        """Get page counts for all PDFs in parallel"""
        tasks = [
            self._get_page_count_async(file_path)
            for file_path in file_paths
        ]
        return await asyncio.gather(*tasks)
    
    async def _get_page_count_async(self, file_path: str) -> int:
        """Get page count for a single PDF"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.io_executor,
            self._get_page_count_sync,
            file_path
        )
    
    def _get_page_count_sync(self, file_path: str) -> int:
        """Synchronous page count extraction"""
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting page count for {file_path}: {str(e)}")
            return 0
    
    async def _extract_text_async(self, file_path: str, progress_callback, progress: ProcessingProgress) -> str:
        """Extract text from a single PDF asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Update progress
        progress.current_file = os.path.basename(file_path)
        if progress_callback:
            await progress_callback(progress)
        
        # Run extraction in thread pool
        text = await loop.run_in_executor(
            self.io_executor,
            self._extract_text_sync,
            file_path,
            progress,
            progress_callback
        )
        
        return text
    
    def _extract_text_sync(self, file_path: str, progress: ProcessingProgress, progress_callback) -> str:
        """Synchronous text extraction with progress updates"""
        try:
            doc = fitz.open(file_path)
            pdf_name = os.path.basename(file_path)
            total_pages = len(doc)
            
            clean_pages = []
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Extract text with layout preservation
                text_dict = page.get_text("dict")
                page_text = self._extract_text_with_spacing(text_dict)
                
                # Extract tables
                tables = self._extract_tables_from_page(page)
                if tables:
                    page_text += "\n\n" + "\n\n".join(tables)
                
                # Clean page text
                cleaned_page = self._filter_pdf_noise_improved(page_text, page_num + 1, total_pages)
                if cleaned_page.strip():
                    metadata_header = f"\n[SOURCE: {pdf_name}, PAGE: {page_num + 1}]\n"
                    clean_pages.append(metadata_header + cleaned_page)
                
                # Update progress
                progress.processed_pages += 1
                if progress_callback:
                    # Run callback in thread-safe way
                    try:
                        asyncio.run_coroutine_threadsafe(progress_callback(progress), asyncio.get_event_loop())
                    except:
                        pass  # Ignore callback errors
            
            doc.close()
            return "\n\n".join(clean_pages)
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    async def _chunk_text_async(self, text: str, progress_callback, progress: ProcessingProgress) -> List[Dict]:
        """Chunk text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor,
            self._chunk_text_sync,
            text,
            progress,
            progress_callback
        )
    
    def _chunk_text_sync(self, text: str, progress: ProcessingProgress, progress_callback) -> List[Dict]:
        """Synchronous text chunking"""
        try:
            # Split text into sentences first
            sentences = self._split_into_sentences(text)
            
            chunks = []
            current_chunk = ""
            current_tokens = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    chunks.append({
                        "id": chunk_id,
                        "text": current_chunk.strip(),
                        "token_count": current_tokens,
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens
                
                # Update progress
                progress.processed_chunks = len(chunks)
                if progress_callback:
                    try:
                        asyncio.run_coroutine_threadsafe(progress_callback(progress), asyncio.get_event_loop())
                    except:
                        pass
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "token_count": current_tokens,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []
    
    def _combine_texts(self, texts: List[str], file_paths: List[str]) -> str:
        """Combine extracted texts from multiple PDFs"""
        combined_parts = []
        for i, text in enumerate(texts):
            if text.strip():
                file_name = os.path.basename(file_paths[i])
                combined_parts.append(f"\n\n--- {file_name} ---\n\n{text}")
        return "\n".join(combined_parts)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with NLP libraries)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        if len(words) <= overlap_tokens:
            return text
        
        overlap_words = words[-overlap_tokens:]
        return " ".join(overlap_words)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback to word count approximation
            return len(text.split()) * 1.3
    
    # Reuse methods from original PDFProcessor
    def _extract_text_with_spacing(self, text_dict):
        """Extract text with proper spacing preserved using layout information"""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_lines = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text:
                            line_text += span_text
                    if line_text.strip():
                        block_lines.append(line_text)
                if block_lines:
                    lines.extend(block_lines)
        
        return "\n".join(lines)
    
    def _extract_tables_from_page(self, page):
        """Extract tables from a page"""
        try:
            tables = page.find_tables()
            table_texts = []
            for table in tables:
                table_data = table.extract()
                if table_data:
                    # Convert table to readable text
                    table_text = "\n".join(["\t".join(row) for row in table_data])
                    table_texts.append(table_text)
            return table_texts
        except:
            return []
    
    def _filter_pdf_noise_improved(self, text: str, page_num: int, total_pages: int) -> str:
        """Improved PDF noise filtering"""
        if not text.strip():
            return ""
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove headers/footers (repeated text)
        lines = text.split('\n')
        filtered_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 3:
                continue
            if line_clean.lower() in ['page', 'of', 'chapter', 'section']:
                continue
            if line_clean.isdigit() and len(line_clean) < 4:
                continue
            if line_clean not in seen_lines or len(line_clean) > 20:
                filtered_lines.append(line)
                seen_lines.add(line_clean)
        
        return '\n'.join(filtered_lines)
