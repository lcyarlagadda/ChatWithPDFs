"""PDF processing service with async support"""

import asyncio
import fitz
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import logging

from utils.text_processor import TextProcessor
from utils.tokenizer import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingProgress:
    """Track processing progress"""
    total_pages: int
    processed_pages: int
    total_chunks: int
    processed_chunks: int
    current_file: str
    status: str
    start_time: float


class PDFService:
    """Service for PDF text extraction and chunking"""
    
    def __init__(self, max_workers: int = 8, chunk_size: int = 500, overlap: int = 50):
        cpu_count = os.cpu_count() or 4
        self.max_workers = min(max_workers, cpu_count, 8)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.text_processor = TextProcessor()
        self.token_counter = TokenCounter()
        
        logger.info(f"Initialized PDFService with {self.max_workers} workers")
    
    async def process_pdfs(self, file_paths: List[str], chunk_size: Optional[int] = None) -> Dict:
        """Process multiple PDFs in parallel"""
        start_time = time.time()
        effective_chunk_size = chunk_size or self.chunk_size
        
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
            # Get page counts
            page_counts = await self._get_page_counts(file_paths)
            progress.total_pages = sum(page_counts)
            
            # Extract text from all PDFs in parallel
            extraction_tasks = [
                self._extract_text_async(file_path, progress)
                for file_path in file_paths
            ]
            extracted_texts = await asyncio.gather(*extraction_tasks)
            
            # Combine and chunk
            progress.status = "chunking"
            combined_text = self._combine_texts(extracted_texts, file_paths)
            chunks = await self._chunk_text_async(combined_text, effective_chunk_size, progress)
            
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
            logger.error(f"Error processing PDFs: {str(e)}")
            raise
        finally:
            self.executor.shutdown(wait=False)
    
    async def _get_page_counts(self, file_paths: List[str]) -> List[int]:
        """Get page counts for all PDFs"""
        tasks = [self._get_page_count_async(path) for path in file_paths]
        return await asyncio.gather(*tasks)
    
    async def _get_page_count_async(self, file_path: str) -> int:
        """Get page count asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_page_count_sync,
            file_path
        )
    
    def _get_page_count_sync(self, file_path: str) -> int:
        """Get page count synchronously"""
        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0
    
    async def _extract_text_async(self, file_path: str, progress: ProcessingProgress) -> str:
        """Extract text asynchronously"""
        progress.current_file = os.path.basename(file_path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_text_sync,
            file_path,
            progress
        )
    
    def _extract_text_sync(self, file_path: str, progress: ProcessingProgress) -> str:
        """Extract text synchronously"""
        try:
            doc = fitz.open(file_path)
            pdf_name = os.path.basename(file_path)
            total_pages = len(doc)
            clean_pages = []
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")
                page_text = self._extract_text_with_spacing(text_dict)
                
                tables = self._extract_tables(page)
                if tables:
                    page_text += "\n\n" + "\n\n".join(tables)
                
                cleaned = self.text_processor.filter_noise(page_text, page_num + 1, total_pages)
                if cleaned.strip():
                    metadata = f"\n[SOURCE: {pdf_name}, PAGE: {page_num + 1}]\n"
                    clean_pages.append(metadata + cleaned)
                
                progress.processed_pages += 1
            
            doc.close()
            return "\n\n".join(clean_pages)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _extract_text_with_spacing(self, text_dict) -> str:
        """Extract text preserving layout"""
        lines = []
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if line_text.strip():
                        lines.append(line_text)
        return "\n".join(lines)
    
    def _extract_tables(self, page) -> List[str]:
        """Extract tables from page"""
        try:
            tables = page.find_tables()
            table_texts = []
            for table in tables:
                table_data = table.extract()
                if table_data:
                    table_text = "\n".join(["\t".join(str(cell or "") for cell in row) for row in table_data])
                    table_texts.append(table_text)
            return table_texts
        except:
            return []
    
    async def _chunk_text_async(self, text: str, chunk_size: int, progress: ProcessingProgress) -> List[Dict]:
        """Chunk text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._chunk_text_sync,
            text,
            chunk_size,
            progress
        )
    
    def _chunk_text_sync(self, text: str, chunk_size: int, progress: ProcessingProgress) -> List[Dict]:
        """Chunk text synchronously with metadata"""
        text = self.text_processor.clean_text(text)
        sentences = self.text_processor.split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count(sentence)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                clean_chunk = current_chunk.strip()
                doc_name, page_num = self.text_processor.extract_metadata(clean_chunk)
                
                chunks.append({
                    "id": chunk_id,
                    "text": clean_chunk,
                    "token_count": current_tokens,
                    "char_count": len(clean_chunk),
                    "document_name": doc_name,
                    "page_number": page_num,
                    "citation": f"[{doc_name}:{page_num}]"
                })
                chunk_id += 1
                
                # Overlap
                words = current_chunk.split()
                overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_tokens = self.token_counter.count(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            
            progress.processed_chunks = len(chunks)
        
        # Final chunk
        if current_chunk.strip():
            clean_chunk = current_chunk.strip()
            doc_name, page_num = self.text_processor.extract_metadata(clean_chunk)
            chunks.append({
                "id": chunk_id,
                "text": clean_chunk,
                "token_count": current_tokens,
                "char_count": len(clean_chunk),
                "document_name": doc_name,
                "page_number": page_num,
                "citation": f"[{doc_name}:{page_num}]"
            })
        
        return chunks
    
    def _combine_texts(self, texts: List[str], file_paths: List[str]) -> str:
        """Combine extracted texts"""
        combined = []
        for i, text in enumerate(texts):
            if text.strip():
                file_name = os.path.basename(file_paths[i])
                combined.append(f"\n\n--- {file_name} ---\n\n{text}")
        return "\n".join(combined)

