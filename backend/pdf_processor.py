import fitz
import re
import os
import tiktoken
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def extract_clean_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with improved spacing and table handling with metadata"""
        doc = fitz.open(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        
        clean_pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Method 1: Extract text with layout preservation
            text_dict = page.get_text("dict")
            page_text = self.extract_text_with_spacing(text_dict)
            
            # Method 2: Also try to extract tables
            tables = self.extract_tables_from_page(page)
            
            # Combine text and tables
            if tables:
                page_text += "\n\n" + "\n\n".join(tables)
            
            # Clean this specific page
            cleaned_page = self.filter_pdf_noise_improved(page_text, page_num + 1, len(doc))
            if cleaned_page.strip():
                # Add metadata prefix to each page
                metadata_header = f"\n[SOURCE: {pdf_name}, PAGE: {page_num + 1}]\n"
                clean_pages.append(metadata_header + cleaned_page)
        
        doc.close()
        return "\n\n".join(clean_pages)

    def extract_text_with_spacing(self, text_dict):
        """Extract text with proper spacing preserved using layout information"""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_lines = []
                prev_y = None
                
                for line in block.get("lines", []):
                    line_text = ""
                    prev_x = None
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        x0 = span.get("bbox", [0])[0]
                        
                        # Add space if there's significant horizontal gap
                        if prev_x is not None and x0 - prev_x > 5:
                            line_text += " "
                        
                        line_text += text
                        prev_x = span.get("bbox", [2])[2]  # Right edge of span
                    
                    if line_text.strip():
                        # Add newline if there's significant vertical gap
                        y0 = line.get("bbox", [0, 0])[1]
                        if prev_y is not None and y0 - prev_y > 15:
                            block_lines.append("")  # Add blank line for paragraph breaks
                        
                        block_lines.append(line_text)
                        prev_y = line.get("bbox", [0, 0, 0, 0])[3]  # Bottom edge
                
                if block_lines:
                    lines.extend(block_lines)
        
        return "\n".join(lines)

    def extract_tables_from_page(self, page):
        """Extract tables from a page and convert to readable text format"""
        tables = []
        
        try:
            # Get tables using PyMuPDF's table extraction
            tabs = page.find_tables()
            
            for table_num, tab in enumerate(tabs):
                if tab and tab.extract():
                    table_data = tab.extract()
                    
                    # Convert table to readable text format
                    table_text = f"[TABLE {table_num + 1}]\n"
                    
                    # Find maximum width for each column
                    col_widths = []
                    for col_idx in range(len(table_data[0]) if table_data else 0):
                        max_width = max(
                            len(str(row[col_idx] or "")) 
                            for row in table_data
                        )
                        col_widths.append(max_width)
                    
                    # Format table rows
                    for row_idx, row in enumerate(table_data):
                        row_text = " | ".join(
                            str(cell or "").ljust(col_widths[col_idx])
                            for col_idx, cell in enumerate(row)
                        )
                        table_text += row_text + "\n"
                        
                        # Add separator after header row
                        if row_idx == 0:
                            separator = "-+-".join("-" * w for w in col_widths)
                            table_text += separator + "\n"
                    
                    tables.append(table_text)
        
        except Exception as e:
            # If table extraction fails, return empty list
            logger.warning(f"Table extraction failed: {e}")
        
        return tables

    def filter_pdf_noise_improved(self, text: str, page_num: int, total_pages: int) -> str:
        """Improved PDF noise filtering - less aggressive, preserves meaningful content"""
        lines = text.split('\n')
        filtered_lines = []
        
        # More targeted header/footer patterns (less aggressive)
        noise_patterns = [
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*Page\s+\d+\s+of\s+\d+\s*$',  # "Page X of Y"
            r'^\s*\d+\s*/\s*\d+\s*$',  # "X / Y"
            r'^\s*©\s*\d{4}.*$',  # Copyright with year
            r'^\s*Copyright\s*©.*\d{4}.*$',  # Copyright notices
            r'^\s*www\.[a-zA-Z0-9-]+\.[a-z]{2,}\s*$',  # Standalone URLs
            r'^\s*https?://[^\s]+\s*$',  # Standalone URLs
        ]
        
        # Keep track of seen lines for duplicate detection
        seen_lines = {}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip completely empty lines
            if not line_stripped:
                # But preserve blank lines for paragraph breaks
                if filtered_lines and filtered_lines[-1] != "":
                    filtered_lines.append("")
                continue
            
            # Skip very short lines (likely noise) - but be less aggressive
            if len(line_stripped) < 2:
                continue
            
            # Check against noise patterns
            is_noise = any(re.match(pattern, line_stripped, re.IGNORECASE) 
                          for pattern in noise_patterns)
            if is_noise:
                continue
            
            # Skip lines repeated more than 3 times (likely headers/footers)
            # But keep lines that appear 2-3 times (could be section headers)
            seen_lines[line_stripped] = seen_lines.get(line_stripped, 0) + 1
            if seen_lines[line_stripped] > 3:
                continue
            
            # Skip lines that are mostly symbols/numbers but preserve some structure
            alpha_chars = len(re.findall(r'[a-zA-Z]', line_stripped))
            total_chars = len(line_stripped)
            if total_chars > 0 and alpha_chars / total_chars < 0.2:
                # Exception: Keep if it looks like a table row or data
                if not any(sep in line_stripped for sep in ['|', '\t', '  ']):
                    continue
            
            # Keep the line with original spacing
            filtered_lines.append(line)
        
        # Join lines with proper spacing
        result = []
        for i, line in enumerate(filtered_lines):
            # If line ends with hyphen, it might be a word break
            if line.rstrip().endswith('-') and i < len(filtered_lines) - 1:
                next_line = filtered_lines[i + 1].strip()
                if next_line and next_line[0].islower():
                    # Remove hyphen and join without space
                    result.append(line.rstrip()[:-1])
                    continue
            
            result.append(line)
        
        # Join with newlines and normalize excessive spacing
        text_result = '\n'.join(result)
        
        # Normalize multiple spaces to single space (but preserve intentional spacing)
        text_result = re.sub(r' {3,}', ' ', text_result)  # Only 3+ spaces
        
        # Normalize excessive newlines
        text_result = re.sub(r'\n{4,}', '\n\n', text_result)
        
        return text_result.strip()

    def clean_text(self, text):
        """Improved text cleaning - preserve spacing and structure"""
        # Fix common PDF extraction issues
        
        # Fix words that got concatenated (e.g., "HelloWorld" -> "Hello World")
        # Look for lowercase followed by uppercase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Normalize multiple spaces (but keep at least one)
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize excessive newlines (keep paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'-\n([a-z])', r'\1', text)
        
        # Ensure space after periods if missing
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Ensure space after commas if missing
        text = re.sub(r',([a-zA-Z])', r', \1', text)
        
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Improved chunking with metadata preservation for citations"""
        
        # First, apply improved text cleaning
        text = self.clean_text(text)
        
        # Extract document name and page info from metadata headers
        def extract_metadata(chunk_text):
            """Extract source document and page number from chunk"""
            source_match = re.search(r'\[SOURCE: ([^,]+), PAGE: (\d+)\]', chunk_text)
            if source_match:
                return source_match.group(1), int(source_match.group(2))
            return "Unknown", 0
        
        # Split by sentences, preserving punctuation
        sentence_endings = r'([.!?]+[\s\n]+)'
        parts = re.split(sentence_endings, text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if parts[i].strip():
                # Combine sentence with its punctuation
                sentence = parts[i]
                if i + 1 < len(parts):
                    sentence += parts[i + 1].rstrip()
                sentences.append(sentence)
        
        # Handle last part if exists
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Test if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Use token count for accurate sizing
            if self.count_tokens(test_chunk) > chunk_size and current_chunk:
                # Save current chunk with metadata
                clean_chunk = current_chunk.strip()
                
                # Extract metadata for this chunk
                doc_name, page_num = extract_metadata(clean_chunk)
                
                chunks.append({
                    "id": chunk_id,
                    "text": clean_chunk,
                    "token_count": self.count_tokens(clean_chunk),
                    "char_count": len(clean_chunk),
                    "document_name": doc_name,
                    "page_number": page_num,
                    "citation": f"[{doc_name}:{page_num}]"
                })
                chunk_id += 1
                
                # Create overlap
                sentences_in_chunk = current_chunk.split('. ')
                if len(sentences_in_chunk) > 1:
                    # Take last sentence for overlap
                    overlap_text = sentences_in_chunk[-1]
                else:
                    # Take last N characters
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = test_chunk
        
        # Add final chunk with metadata
        if current_chunk.strip():
            clean_chunk = current_chunk.strip()
            doc_name, page_num = extract_metadata(clean_chunk)
            
            chunks.append({
                "id": chunk_id,
                "text": clean_chunk,
                "token_count": self.count_tokens(clean_chunk),
                "char_count": len(clean_chunk),
                "document_name": doc_name,
                "page_number": page_num,
                "citation": f"[{doc_name}:{page_num}]"
            })
        
        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

