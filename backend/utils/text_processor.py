"""Text processing utilities for cleaning and chunking"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text cleaning, noise filtering, and sentence splitting"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by fixing common PDF extraction issues"""
        # Fix concatenated words (e.g., "HelloWorld" -> "Hello World")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'-\n([a-z])', r'\1', text)
        
        # Ensure space after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Ensure space after commas
        text = re.sub(r',([a-zA-Z])', r', \1', text)
        
        return text.strip()
    
    @staticmethod
    def filter_noise(text: str, page_num: int, total_pages: int) -> str:
        """Filter PDF noise (headers, footers, page numbers)"""
        lines = text.split('\n')
        filtered_lines = []
        
        noise_patterns = [
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*Page\s+\d+\s+of\s+\d+\s*$',
            r'^\s*\d+\s*/\s*\d+\s*$',
            r'^\s*©\s*\d{4}.*$',
            r'^\s*Copyright\s*©.*\d{4}.*$',
            r'^\s*www\.[a-zA-Z0-9-]+\.[a-z]{2,}\s*$',
            r'^\s*https?://[^\s]+\s*$',
        ]
        
        seen_lines = {}
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                if filtered_lines and filtered_lines[-1] != "":
                    filtered_lines.append("")
                continue
            
            if len(line_stripped) < 2:
                continue
            
            is_noise = any(re.match(pattern, line_stripped, re.IGNORECASE) 
                          for pattern in noise_patterns)
            if is_noise:
                continue
            
            seen_lines[line_stripped] = seen_lines.get(line_stripped, 0) + 1
            if seen_lines[line_stripped] > 3:
                continue
            
            alpha_chars = len(re.findall(r'[a-zA-Z]', line_stripped))
            total_chars = len(line_stripped)
            if total_chars > 0 and alpha_chars / total_chars < 0.2:
                if not any(sep in line_stripped for sep in ['|', '\t', '  ']):
                    continue
            
            filtered_lines.append(line)
        
        # Fix hyphenated words at line breaks
        result = []
        for i, line in enumerate(filtered_lines):
            if line.rstrip().endswith('-') and i < len(filtered_lines) - 1:
                next_line = filtered_lines[i + 1].strip()
                if next_line and next_line[0].islower():
                    result.append(line.rstrip()[:-1])
                    continue
            result.append(line)
        
        text_result = '\n'.join(result)
        text_result = re.sub(r' {3,}', ' ', text_result)
        text_result = re.sub(r'\n{4,}', '\n\n', text_result)
        
        return text_result.strip()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def extract_metadata(chunk_text: str) -> tuple[str, int]:
        """Extract document name and page number from chunk"""
        source_match = re.search(r'\[SOURCE: ([^,]+), PAGE: (\d+)\]', chunk_text)
        if source_match:
            return source_match.group(1), int(source_match.group(2))
        return "Unknown", 0

