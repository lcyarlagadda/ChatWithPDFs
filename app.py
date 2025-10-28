import streamlit as st
import pandas as pd
import fitz
import re
import os
import tiktoken
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from rank_bm25 import BM25Okapi
import json
import time

import logging
from datetime import datetime
from pathlib import Path

import traceback

st.set_page_config(page_title="Clean PDF Chat", layout="wide")

st.markdown("""
<style>
    /* Remove upload zone border */
    .upload-zone {
        border: none;
        padding: 0;
        margin: 0;
    }
    
    /* Remove file uploader border completely */
    .stFileUploader > div {
        border: none !important;
    }
    
    .stFileUploader > div > div {
        border: none !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* User messages - right aligned */
    .user-message {
        background: #007bff;
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 10px 0;
        text-align: right;
        margin-left: 20%;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    /* Bot messages - left aligned */
    .bot-message {
        background: #f8f9fa;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
        margin-right: 20%;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    /* Context boxes - better visibility */
    .context-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #333;
        border: 1px solid #dee2e6;
    }
    
    /* Citations - dark theme for visibility */
    .citation {
        background: #343a40;
        color: #ffffff;
        border: 1px solid #495057;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
        font-weight: 500;
    }
    
    /* Professional slider styling - subtle colors */
    .stSlider > div > div > div {
        background: #e9ecef !important;
        border-radius: 10px !important;
        height: 8px !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #6c757d, #adb5bd) !important;
        border-radius: 10px !important;
        height: 8px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: #495057 !important;
        border: none !important;
        border-radius: 50% !important;
        width: 16px !important;
        height: 16px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    .stSlider > div > div > div > div > div:hover {
        background: #212529 !important;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stSlider > div > div > div > div > div:active {
        background: #0d0f12 !important;
    }
    
    /* Remove slider value display circles */
    .stSlider > div > div > div > div > div > div {
        display: none !important;
    }
    
    /* Selectbox/Dropdown styling */
    .stSelectbox > div > div > div {
        background-color: white !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
    }
    
    /* Dropdown text color - make it more visible */
    .stSelectbox > div > div > div input {
        color: #333 !important;
        background-color: white !important;
    }
    
    /* Dropdown placeholder text */
    .stSelectbox > div > div > div [data-baseweb="select"] {
        color: #333 !important;
    }
    
    /* Dropdown option text */
    .stSelectbox ul li {
        color: #333 !important;
        background-color: white !important;
    }
    
    /* Dropdown selected value */
    .stSelectbox > div > div > div > div[data-baseweb="select"] > div {
        color: #333 !important;
        background-color: white !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        color: #333 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
        outline: none !important;
    }
    
    /* Checkbox styling - subtle gray/blue */
    .stCheckbox > div > label > div[data-baseweb="checkbox"] {
        background-color: white !important;
        border: 2px solid #adb5bd !important;
    }
    
    .stCheckbox > div > label > div[data-baseweb="checkbox"]:hover {
        border-color: #6c757d !important;
    }
    
    .stCheckbox > div > label > input:checked + div[data-baseweb="checkbox"] {
        background-color: #007bff !important;
        border-color: #007bff !important;
    }
    
    /* Checkbox checkmark color */
    .stCheckbox > div > label > input:checked + div[data-baseweb="checkbox"] svg {
        color: white !important;
    }
    
    /* Checkbox text color */
    .stCheckbox > div > label > div[data-testid="stMarkdownContainer"] {
        color: #333 !important;
    }
    
    /* Better button styling */
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #0056b3;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.3);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Better spacing - minimal top padding */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
    }
    
    /* Remove extra header padding */
    .stApp > div:first-child {
        padding-top: 0;
    }
    
    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Remove any extra margins from title */
    .stApp h1 {
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Ensure clean top spacing */
    .stApp > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)



def extract_clean_text_from_pdf(pdf_path):
    """Extract text from PDF with improved spacing and table handling with metadata"""
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    
    clean_pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Method 1: Extract text with layout preservation
        text_dict = page.get_text("dict")
        page_text = extract_text_with_spacing(text_dict)
        
        # Method 2: Also try to extract tables
        tables = extract_tables_from_page(page)
        
        # Combine text and tables
        if tables:
            page_text += "\n\n" + "\n\n".join(tables)
        
        # Clean this specific page
        cleaned_page = filter_pdf_noise_improved(page_text, page_num + 1, len(doc))
        if cleaned_page.strip():
            # Add metadata prefix to each page
            metadata_header = f"\n[SOURCE: {pdf_name}, PAGE: {page_num + 1}]\n"
            clean_pages.append(metadata_header + cleaned_page)
    
    doc.close()
    return "\n\n".join(clean_pages)


def extract_text_with_spacing(text_dict):
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


def extract_tables_from_page(page):
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
        pass
    
    return tables


def filter_pdf_noise_improved(text: str, page_num: int, total_pages: int) -> str:
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


def clean_text(text):
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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Improved chunking with metadata preservation for citations"""
    
    # First, apply improved text cleaning
    text = clean_text(text)
    
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
        if count_tokens(test_chunk) > chunk_size and current_chunk:
            # Save current chunk with metadata
            clean_chunk = current_chunk.strip()
            
            # Extract metadata for this chunk
            doc_name, page_num = extract_metadata(clean_chunk)
            
            chunks.append({
                "id": chunk_id,
                "text": clean_chunk,
                "token_count": count_tokens(clean_chunk),
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
            "token_count": count_tokens(clean_chunk),
            "char_count": len(clean_chunk),
            "document_name": doc_name,
            "page_number": page_num,
            "citation": f"[{doc_name}:{page_num}]"
        })
    
    return chunks


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def optimize_context_window(chunks: List[Dict], question: str, max_tokens: int = 1800) -> List[Dict]:
    """
    Dynamically select chunks within token limit
    Prioritizes highest-scoring chunks that fit within context window
    """
    # Calculate tokens for prompt structure
    prompt_overhead = 150  # Approximate tokens for prompt template
    available_tokens = max_tokens - prompt_overhead
    
    selected_chunks = []
    total_tokens = 0
    
    # Sort chunks by score (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x.get('final_score', x.get('combined_score', 0)), reverse=True)
    
    for chunk in sorted_chunks:
        chunk_tokens = chunk.get('token_count', count_tokens(chunk['text']))
        
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

def display_query_metrics(metrics: Dict, retrieved_chunks: List[Dict] = None):
    """Display comprehensive query metrics in an organized format"""
    
    # Create metrics container
    with st.expander("Query Performance Metrics", expanded=False):
        # Timing metrics
        st.markdown("### Timing Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Latency", 
                f"{metrics.get('total_latency', 0):.2f}s",
                help="End-to-end query processing time"
            )
        
        with col2:
            st.metric(
                "Retrieval Time", 
                f"{metrics.get('retrieval_time', 0):.2f}s",
                help="Time to retrieve relevant chunks"
            )
        
        with col3:
            st.metric(
                "Generation Time", 
                f"{metrics.get('generation_time', 0):.2f}s",
                help="Time to generate the response"
            )
        
        # Token metrics
        st.markdown("### Token Usage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Input Tokens", 
                f"{metrics.get('input_tokens', 0):,}",
                help="Tokens in prompt (context + question)"
            )
        
        with col2:
            st.metric(
                "Output Tokens", 
                f"{metrics.get('output_tokens', 0):,}",
                help="Tokens generated in response"
            )
        
        with col3:
            total_tokens = metrics.get('input_tokens', 0) + metrics.get('output_tokens', 0)
            st.metric(
                "Total Tokens", 
                f"{total_tokens:,}",
                help="Combined input + output tokens"
            )
        
        # Context information
        st.markdown("### Context Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Context Length", 
                f"{metrics.get('context_length', 0):,} chars",
                help="Total characters in retrieved context"
            )
        
        with col2:
            st.metric(
                "Response Length", 
                f"{metrics.get('response_length', 0):,} chars",
                help="Total characters in generated response"
            )
        
        # Performance analysis
        if metrics.get('total_latency', 0) > 0:
            st.markdown("### Performance Analysis")
            
            retrieval_pct = (metrics.get('retrieval_time', 0) / metrics.get('total_latency', 1)) * 100
            generation_pct = (metrics.get('generation_time', 0) / metrics.get('total_latency', 1)) * 100
            overhead_pct = 100 - retrieval_pct - generation_pct
            
            # Create a simple progress bar visualization
            st.markdown(f"""
            <div style="background: #f0f2f6; border-radius: 10px; padding: 10px; margin: 10px 0;">
                <div style="display: flex; height: 30px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #4CAF50; width: {retrieval_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {retrieval_pct:.1f}%
                    </div>
                    <div style="background: #2196F3; width: {generation_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {generation_pct:.1f}%
                    </div>
                    <div style="background: #9E9E9E; width: {overhead_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {overhead_pct:.1f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.9em;">
                    <span>🟢 Retrieval</span>
                    <span>🔵 Generation</span>
                    <span>⚪ Overhead</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bottleneck identification
            if retrieval_pct > 60:
                st.warning("⚠️ **Retrieval Bottleneck**: Consider reducing chunk count or optimizing index")
            elif generation_pct > 60:
                st.warning("⚠️ **Generation Bottleneck**: Consider reducing max_tokens or using smaller model")
            else:
                st.success("✅ **Balanced Performance**: Good distribution between retrieval and generation")
        
        # Chunk details
        if retrieved_chunks:
            st.markdown("### Retrieved Chunks Details")
            
            # Create a table of chunk scores
            chunk_data = []
            for chunk in retrieved_chunks[:5]:  # Show top 5
                chunk_data.append({
                    "Rank": chunk.get('rank', '-'),
                    "Score": f"{chunk.get('final_score', 0):.3f}",
                    "Dense": f"{chunk.get('dense_score', 0):.3f}",
                    "Sparse": f"{chunk.get('sparse_score', 0):.3f}",
                    "Length": f"{len(chunk.get('text', ''))} chars"
                })
            
            df = pd.DataFrame(chunk_data)
            st.dataframe(df, use_container_width=True)
            
            # Show text preview of top chunk
            if retrieved_chunks:
                st.markdown("**Top Chunk Preview:**")
                preview_text = retrieved_chunks[0].get('text', '')[:200]
                st.code(preview_text + "..." if len(preview_text) == 200 else preview_text)



# Model Manager for lazy loading
class SimpleModelManager:
    def __init__(self):
        self._embedding_model = None
        self._reranker = None
        self._mistral_model = None
        self._mistral_tokenizer = None
    
    @st.cache_resource(show_spinner=False)
    def get_embedding_model(_self):
        if _self._embedding_model is None:
            _self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _self._embedding_model
    
    @st.cache_resource(show_spinner=False)
    def get_reranker(_self):
        if _self._reranker is None:
            _self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return _self._reranker
    
    @st.cache_resource(show_spinner=False)
    def get_mistral_model(_self):
        if _self._mistral_model is None or _self._mistral_tokenizer is None:
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                raise ValueError("Hugging Face token not found")
            
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
            
            _self._mistral_model = model
            _self._mistral_tokenizer = tokenizer
        return _self._mistral_model, _self._mistral_tokenizer


# Simple Retriever
class SimpleRetriever:
    def __init__(self, embedding_model, faiss_index, bm25_index, metadata, reranker=None):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.metadata = metadata
        self.reranker = reranker
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        # Dense retrieval
        query_embedding = self.embedding_model.encode([query])[0]
        distances, dense_indices = self.faiss_index.search(query_embedding.reshape(1, -1).astype('float32'), k*2)
        
        # Sparse retrieval
        bm25_scores = self.bm25_index.get_scores(query.split())
        sparse_indices = np.argsort(bm25_scores)[::-1][:k*2]
        
        # Combine results
        all_indices = list(set(dense_indices[0].tolist() + sparse_indices.tolist()))
        
        chunks_with_scores = []
        for idx in all_indices:
            if idx < len(self.metadata["chunks"]):
                chunk = self.metadata["chunks"][idx].copy()
                
                # Calculate scores
                dense_score = 0
                if idx in dense_indices[0]:
                    dense_idx = np.where(dense_indices[0] == idx)[0]
                    if len(dense_idx) > 0:
                        dense_score = 1 / (1 + distances[0][dense_idx[0]])
                
                sparse_score = bm25_scores[idx] / (np.max(bm25_scores) + 1e-8)
                combined_score = 0.6 * dense_score + 0.4 * sparse_score
                
                # Round scores to 3 decimal places
                chunk["dense_score"] = round(dense_score, 3)
                chunk["sparse_score"] = round(sparse_score, 3)
                chunk["combined_score"] = round(combined_score, 3)
                chunks_with_scores.append(chunk)
        
        # Sort by combined score
        chunks_with_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Rerank if available
        if self.reranker and len(chunks_with_scores) > k:
            query_chunk_pairs = [(query, chunk["text"]) for chunk in chunks_with_scores[:k*2]]
            rerank_scores = self.reranker.predict(query_chunk_pairs)
            
            for i, chunk in enumerate(chunks_with_scores[:k*2]):
                chunk["rerank_score"] = round(rerank_scores[i], 3)
                chunk["final_score"] = round(0.7 * chunk["combined_score"] + 0.3 * rerank_scores[i], 3)
        else:
            for chunk in chunks_with_scores:
                chunk["final_score"] = chunk["combined_score"]  # Already rounded
        
        chunks_with_scores.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add ranking
        for i, chunk in enumerate(chunks_with_scores[:k]):
            chunk["rank"] = i + 1
        
        return chunks_with_scores[:k]

# Simple RAG System
class SimpleRAGSystem:
    def __init__(self, retriever, model, tokenizer):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
    
    def answer_question(self, question: str, k: int = 3) -> Dict:
        query_start_time = time.time()  # Start query timing
        
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
        retrieved_chunks = optimize_context_window(retrieved_chunks, question, max_tokens=1800)
        
        #  Show retrieved chunks - only show highly relevant chunks (score > 0.3)
        if "show_chunks" in st.session_state and st.session_state.show_chunks:
            # Filter for highly relevant chunks
            relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.get('final_score', 0) > 0.3]
            
            with st.expander("🔍 Debug: Retrieved Chunks", expanded=True):
                if relevant_chunks:
                    # Show only the most relevant chunk(s) - at most 2
                    chunks_to_show = relevant_chunks[:2] if len(relevant_chunks) > 1 else relevant_chunks[:1]
                    
                    for i, chunk in enumerate(chunks_to_show):
                        st.write(f"**Retrieved Chunk {i+1} (Score: {chunk.get('final_score', 0)})**")
                        st.code(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                        if i < len(chunks_to_show) - 1:
                            st.write("---")
                else:
                    # If no highly relevant chunks, show the top one anyway
                    if retrieved_chunks:
                        st.write(f"**Retrieved Chunk (Score: {retrieved_chunks[0].get('final_score', 0)})**")
                        st.code(retrieved_chunks[0]['text'][:300] + "..." if len(retrieved_chunks[0]['text']) > 300 else retrieved_chunks[0]['text'])
                        st.caption("*Note: No chunks with high relevance score (> 0.3) found. Showing top result anyway.*")
        
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
        
        # Use streaming response for better UX
        message_placeholder = st.empty()  # Placeholder for streaming output
        generation_result = self.generate_response_streaming(rag_prompt, max_length=512, message_placeholder=message_placeholder)
        response = generation_result.get("response", "")
        response_metrics = generation_result  # Store metrics

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
        
        # Log the query
        query_logger.log_query(question, response, metrics, retrieved_chunks)
        
        return {
            "question": question, 
            "answer": response, 
            "retrieved_chunks": retrieved_chunks, 
            "context": context,
            "is_general_question": is_general_question,
            "retrieval_count": len(retrieved_chunks)
        }
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
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
        
        return response

    def generate_response_streaming(self, prompt: str, max_length: int = 512, message_placeholder=None):
        """Generate response with TRUE token-level streaming for real-time output"""
        generation_start = time.time()
        token_count = 0
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        full_response = ""
        
        # Generate tokens one at a time with streaming
        with torch.no_grad():
            # Start with the input
            generated_ids = inputs['input_ids']
            past_key_values = None
            
            for _ in range(max_length):
                # Generate next token
                if past_key_values is None:
                    # First iteration - use full input
                    outputs = self.model(
                        input_ids=generated_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    # Subsequent iterations - only use last token
                    outputs = self.model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                past_key_values = outputs.past_key_values
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-p sampling
                next_token_logits = next_token_logits / 0.2  # temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > 0.85  # top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we hit EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Decode only the new token
                new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Skip empty tokens
                if new_text.strip() == '':
                    continue
                
                full_response += new_text
                token_count += 1
                
                # Update UI in real-time
                if message_placeholder is not None:
                    # Clean response before displaying
                    display_text = full_response
                    
                    # Remove prompt artifacts during streaming
                    if "[/INST]" in display_text:
                        display_text = display_text.split("[/INST]")[-1].strip()
                    if "CONTEXT:" in display_text:
                        display_text = display_text.split("CONTEXT:")[0].strip()
                    if "QUESTION:" in display_text:
                        display_text = display_text.split("QUESTION:")[0].strip()
                    
                    message_placeholder.markdown(
                        f'<div class="bot-message">{display_text}▌</div>',  # Add cursor
                        unsafe_allow_html=True
                    )
                
                # Small delay for smoother streaming visualization
                time.sleep(0.01)
        
        # Final cleanup
        if prompt in full_response:
            full_response = full_response.split(prompt)[-1].strip()
        
        if "[/INST]" in full_response:
            full_response = full_response.split("[/INST]")[-1].strip()
        
        if "CONTEXT:" in full_response:
            full_response = full_response.split("CONTEXT:")[0].strip()
        if "QUESTION:" in full_response:
            full_response = full_response.split("QUESTION:")[0].strip()
        
        full_response = full_response.replace("[INST]", "").replace("[/INST]", "").strip()
        
        # Update final display without cursor
        if message_placeholder is not None:
            message_placeholder.markdown(
                f'<div class="bot-message">{full_response}</div>',
                unsafe_allow_html=True
            )
        
        # Log generation metrics
        generation_time = time.time() - generation_start
        input_tokens = input_length
        
        return {
            "response": full_response,
            "input_tokens": input_tokens,
            "output_tokens": token_count,
            "generation_time": generation_time
        }

class QueryLogger:
    """Enhanced logger with chunk tracking, cost calculation, and traces"""
    
    def __init__(self):
        self.query_history = []
        self.chunk_tracking = []  # Track which chunks were retrieved
        self.traces = []  # Store detailed traces for debugging
    
    def log_query(self, query: str, response: str, metrics: Dict, retrieved_chunks=None):
        """Log a query with enhanced metrics including chunks and costs"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_length': len(response),
            'metrics': metrics,
            'retrieved_chunks': self._serialize_chunks(retrieved_chunks),
            'chunk_count': len(retrieved_chunks) if retrieved_chunks else 0
        }
        
        # Calculate cost
        cost = self._calculate_cost(metrics.get('input_tokens', 0), metrics.get('output_tokens', 0))
        log_entry['metrics']['cost_usd'] = cost
        
        self.query_history.append(log_entry)
        
        # Log to file and console
        logger.info(f"Query logged: {query[:50]}...")
        logger.info(f"  Latency: {metrics.get('total_latency', 0):.2f}s")
        logger.info(f"  Tokens: {metrics.get('total_tokens', 0)} (in: {metrics.get('input_tokens', 0)}, out: {metrics.get('output_tokens', 0)})")
        logger.info(f"  Chunks retrieved: {len(retrieved_chunks) if retrieved_chunks else 0}")
        logger.info(f"  Cost: ${cost:.6f}")
        
        return log_entry
    
    def log_trace(self, trace_id: str, trace_data: Dict):
        """Store detailed trace for debugging"""
        trace = {
            'trace_id': trace_id,
            'timestamp': datetime.now().isoformat(),
            'data': trace_data
        }
        self.traces.append(trace)
        logger.debug(f"Trace logged: {trace_id}")
    
    def _serialize_chunks(self, chunks):
        """Serialize chunk information for logging"""
        if not chunks:
            return []
        
        serialized = []
        for i, chunk in enumerate(chunks[:10]):  # Log first 10 chunks
            serialized.append({
                'rank': i + 1,
                'score': chunk.get('final_score', chunk.get('combined_score', 0)),
                'length': len(chunk.get('text', '')),
                'text_preview': chunk.get('text', '')[:100]  # First 100 chars
            })
        return serialized
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on Mistral-7B pricing (example rates)"""
        # Mistral-7B pricing (example - adjust based on actual pricing)
        input_cost_per_1k = 0.00025  # $0.25 per 1M input tokens
        output_cost_per_1k = 0.00025  # $0.25 per 1M output tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_stats(self):
        """Get aggregated statistics with cost analysis"""
        if not self.query_history:
            return {}
        
        latencies = [q['metrics'].get('total_latency', 0) for q in self.query_history]
        retrieval_times = [q['metrics'].get('retrieval_time', 0) for q in self.query_history]
        generation_times = [q['metrics'].get('generation_time', 0) for q in self.query_history]
        tokens = [q['metrics'].get('total_tokens', 0) for q in self.query_history]
        costs = [q['metrics'].get('cost_usd', 0) for q in self.query_history]
        chunks = [q.get('chunk_count', 0) for q in self.query_history]
        
        return {
            'total_queries': len(self.query_history),
            'avg_latency': sum(latencies) / len(latencies),
            'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            'avg_generation_time': sum(generation_times) / len(generation_times) if generation_times else 0,
            'total_tokens': sum(tokens),
            'avg_tokens_per_query': sum(tokens) / len(tokens),
            'total_cost_usd': sum(costs),
            'avg_cost_per_query': sum(costs) / len(costs),
            'avg_chunks_per_query': sum(chunks) / len(chunks) if chunks else 0
        }
    
    def get_bottleneck_analysis(self):
        """Analyze bottlenecks in retrieval vs generation"""
        if not self.query_history:
            return {}
        
        bottleneck_stats = {
            'retrieval_heavy': 0,
            'generation_heavy': 0,
            'balanced': 0
        }
        
        for query in self.query_history:
            retrieval = query['metrics'].get('retrieval_time', 0)
            generation = query['metrics'].get('generation_time', 0)
            total = query['metrics'].get('total_latency', 0)
            
            if total > 0:
                retrieval_pct = (retrieval / total) * 100
                if retrieval_pct > 60:
                    bottleneck_stats['retrieval_heavy'] += 1
                elif generation / total > 60:
                    bottleneck_stats['generation_heavy'] += 1
                else:
                    bottleneck_stats['balanced'] += 1
        
        return bottleneck_stats
    
    def export_traces_json(self, filename: str = None):
        """Export traces to JSON file for external analysis (e.g., Langfuse)"""
        if filename is None:
            filename = f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'queries': self.query_history,
            'traces': self.traces,
            'stats': self.get_stats(),
            'bottlenecks': self.get_bottleneck_analysis(),
            'export_time': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Traces exported to {filename}")
        return filename


# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"chat_log_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize query logger
query_logger = QueryLogger()

# Initialize model manager
model_manager = SimpleModelManager()

# Main UI
st.title("PDF Chat System")

# Initialize session state variables BEFORE sidebar
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "filtering_stats" not in st.session_state:
    st.session_state.filtering_stats = {}
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True
if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = False

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Mistral-7B", "FLAN-T5", "GPT-2"],
        help="Choose the language model for generation"
    )
    
    # Retriever selection
    retriever_type = st.selectbox(
        "Select Retriever",
        ["Hybrid (Dense + Sparse)", "Dense Only", "Sparse Only"],
        help="Choose the retrieval method"
    )
    
    # Number of chunks
    num_chunks = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="More chunks = more context but slower"
    )
    
    # Set default values
    chunk_size = 500
    
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.2)
    max_tokens = st.slider("Max Tokens", 100, 1000, 512)
    
        # noise filtering options
    enable_noise_filtering = st.checkbox("Enable Advanced Noise Filtering", value=True,
                                       help="Remove headers, footers, tables, and other PDF artifacts")
    
    # Debug option
    show_chunks = st.checkbox("Show Retrieved Chunks (Debug)", value=False,
                              help="Display the chunks that were retrieved for each query")
    st.session_state.show_chunks = show_chunks

# Persistent error display in sidebar
if "error_message" in st.session_state and st.session_state.error_message:
    with st.sidebar:
        st.error("Error Occured")
        with st.expander("View Error Details", expanded=True):
            st.code(st.session_state.error_message, language="python")
            st.write("**Error Details:**")
            st.text_area("", value=st.session_state.error_message, height=200, key="error_display", label_visibility="collapsed")
        if st.button("Clear Error"):
            st.session_state.error_message = None
            st.rerun()

# Conversation management in main page
if len(st.session_state.get("conversation", [])) > 0:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Conversation"):
            if "conversation" in st.session_state and len(st.session_state.conversation) > 0:
                conversation_text = f"Conversation Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                conversation_text += "=" * 60 + "\n\n"
                
                for i, msg in enumerate(st.session_state.conversation):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_text += f"{role}:\n{msg['text']}\n\n"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.txt"
                
                st.download_button(
                    label="Download (.txt)",
                    data=conversation_text,
                    file_name=filename,
                    mime="text/plain"
                )
                
                # Also export as JSON for structured data
                serializable_conversation = []
                for msg in st.session_state.conversation:
                    serializable_msg = {
                        "role": msg["role"],
                        "text": msg["text"]
                    }
                    if "context" in msg:
                        serializable_msg["context"] = msg["context"]
                    if "is_general_question" in msg:
                        serializable_msg["is_general_question"] = msg["is_general_question"]
                    if "retrieval_count" in msg:
                        serializable_msg["retrieval_count"] = msg["retrieval_count"]
                    serializable_conversation.append(serializable_msg)
                
                import json
                conversation_json = json.dumps(serializable_conversation, indent=2)
                json_filename = f"conversation_{timestamp}.json"
                
                st.download_button(
                    label="Download (.json)",
                    data=conversation_json,
                    file_name=json_filename,
                    mime="application/json"
                )
    
    with col2:
        if st.button("🗑️ Clear Conversation"):
            if "conversation" in st.session_state:
                st.session_state.conversation = []
                st.session_state.input_counter = 0
            st.rerun()
    
    st.divider()

# Upload zone
uploaded_files = st.file_uploader(
    "📁 Upload PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload one or more PDF files to analyze"
)

# Process uploaded files
if uploaded_files:
    if st.session_state.rag_system is None:
        # Show single loading message
        loading_placeholder = st.info("Processing the file... This may take a few moments.")
        try:
            # Extract clean text from all PDFs
            all_pdf_text = ""
            all_pdf_stats = []
            raw_extracted_texts = []  # Store raw extracted text for debugging
            
            for uploaded_file in uploaded_files:
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract raw text first (without filtering) for debugging
                doc = fitz.open(f"temp_{uploaded_file.name}")
                raw_text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    blocks = page.get_text("blocks")
                    raw_page_text = "\n".join([b[4] for b in blocks]) if blocks else page.get_text("text")
                    raw_text += f"\n--- Page {page_num + 1} ---\n{raw_page_text}\n"
                doc.close()
                raw_extracted_texts.append(raw_text)
                
                pdf_text = extract_clean_text_from_pdf(f"temp_{uploaded_file.name}")
                all_pdf_text += f"\n\n--- {uploaded_file.name} ---\n\n"
                all_pdf_text += pdf_text
                
                os.remove(f"temp_{uploaded_file.name}")
            
            # Chunk the text
            chunks = chunk_text(all_pdf_text, chunk_size=chunk_size)
            
            # Generate embeddings
            embedding_model = model_manager.get_embedding_model()
            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_model.encode(texts)
            
            # Add embeddings to chunks
            chunks_with_embeddings = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = chunk.copy()
                enhanced_chunk["embedding"] = embeddings[i]
                chunks_with_embeddings.append(enhanced_chunk)
            
            # Create FAISS index
            embeddings_array = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
            dimension = embeddings_array.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(embeddings_array.astype('float32'))
            
            # Create BM25 index
            tokenized_texts = [text.split() for text in texts]
            bm25_index = BM25Okapi(tokenized_texts)
            
            # Load reranker
            reranker = model_manager.get_reranker()
            
            # Create retriever
            retriever = SimpleRetriever(embedding_model, faiss_index, bm25_index, {"chunks": chunks_with_embeddings}, reranker)
            
            # Load Mistral model
            mistral_model, mistral_tokenizer = model_manager.get_mistral_model()
            
            # Create RAG system
            st.session_state.rag_system = SimpleRAGSystem(retriever, mistral_model, mistral_tokenizer)
            
            # Store chunks for debugging
            st.session_state.raw_chunks = chunks
            
            # Clear loading message and show success
            loading_placeholder.empty()
            st.success(f" Successfully processed {len(uploaded_files)} PDF(s) with {len(chunks)} clean chunks! You can now ask questions.")
            
            # DEBUG: Show extraction pipeline
            with st.expander("🔍 DEBUG: Text Extraction Pipeline", expanded=True):
                st.write("### 1️⃣ Raw Extracted Text (from PDF)")
                if raw_extracted_texts:
                    # Show first 1000 chars of raw text
                    st.code(raw_extracted_texts[0][:1000] + "..." if len(raw_extracted_texts[0]) > 1000 else raw_extracted_texts[0])
                
                st.write("### 2️⃣ Filtered Text (after noise removal)")
                # Show first 1000 chars of filtered text
                filtered_preview = all_pdf_text[:1000] + "..." if len(all_pdf_text) > 1000 else all_pdf_text
                st.code(filtered_preview)
                
                st.write("### 3️⃣ Chunked Text")
                # Show first 3 chunks
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i+1}:**")
                    st.code(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                    st.write(f"Characters: {len(chunk['text'])}, Tokens: {chunk['token_count']}")
                    st.write("---")
            
            # Show first few chunks for debugging
            with st.expander("🔍 Debug: View First 3 Chunks", expanded=False):
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i+1}:**")
                    st.code(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                    st.write(f"Characters: {len(chunk['text'])}, Tokens: {chunk['token_count']}")
                    st.write("---")
            
        except Exception as e:
            # Clear loading message on error
            loading_placeholder.empty()
            import traceback
            error_trace = traceback.format_exc()
            st.error(f"Error processing PDFs: {str(e)}")
            with st.expander("Error Details", expanded=False):
                st.code(error_trace)
            logger.error(f"Error processing PDFs: {str(e)}\n{error_trace}")
            error_display = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
            st.session_state.error_message = error_display
            st.stop()
    
    # Display conversation summary if exists
    if len(st.session_state.conversation) > 0:
        with st.expander("Conversation Summary", expanded=False):
            st.write(f"**Total Exchanges:** {len(st.session_state.conversation) // 2}")
            st.write(f"**Documents Processed:** {len(uploaded_files)} file(s)")
            if "raw_chunks" in st.session_state and st.session_state.raw_chunks:
                st.write(f"**Total Chunks:** {len(st.session_state.raw_chunks)}")
    
    # Display conversation
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for i, msg in enumerate(st.session_state.conversation):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["text"]}</div>', unsafe_allow_html=True)
            
            # Show citations if available
            if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
                citations = []
                seen_citations = set()
                for chunk in msg["retrieved_chunks"]:
                    citation = chunk.get('citation', '')
                    if citation and citation not in seen_citations:
                        citations.append(citation)
                        seen_citations.add(citation)
                
                if citations:
                    citation_text = " ".join([f'<span class="citation">{c}</span>' for c in citations])
                    st.markdown(f'<div style="margin-top: 8px; margin-bottom: 16px;">{citation_text}</div>', unsafe_allow_html=True)

        if "metrics" in msg and msg["metrics"]:
            display_query_metrics(msg["metrics"], msg.get("retrieved_chunks"))
    
    # Show "Thinking..." while processing
    if st.session_state.processing:
        st.info("Thinking...")
    
    # User input
    user_input_key = f"input_{st.session_state.get('input_counter', 0)}"
    user_input = st.text_input(
        "Ask a question about the PDF:",
        placeholder="Type your question here...",
        disabled=st.session_state.processing,
        key=user_input_key
    )
    
    # Handle input submission
    if user_input and not st.session_state.processing:
        if "last_input" not in st.session_state or st.session_state.last_input != user_input:
            # Add user message
            st.session_state.conversation.append({"role": "user", "text": user_input})
            st.session_state.processing = True
            st.session_state.last_input = user_input
            # Increment counter to force new text input (clears it)
            st.session_state['input_counter'] = st.session_state.get('input_counter', 0) + 1
            st.rerun()
    
    # Process bot response
    if st.session_state.processing and len(st.session_state.conversation) > 0 and st.session_state.conversation[-1]["role"] == "user":
        try:
            user_question = st.session_state.conversation[-1]['text']
            
            # Generate response
            result = st.session_state.rag_system.answer_question(user_question, k=num_chunks)
            
            # Clear the processing state
            st.session_state.processing = False
            
            # Get metrics from the latest log entry
            latest_metrics = {}
            if query_logger.query_history:
                latest_metrics = query_logger.query_history[-1].get('metrics', {})
            
            # Add the response to conversation with metrics
            st.session_state.conversation.append({
                "role": "bot", 
                "text": result["answer"],
                "context": result["context"],
                "retrieved_chunks": result["retrieved_chunks"],
                "is_general_question": result.get("is_general_question", False),
                "retrieval_count": result.get("retrieval_count", 0),
                "metrics": latest_metrics
            })
            
            # Rerun to show the response (metrics will be displayed in the conversation loop)
            st.rerun()
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_display = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
            st.session_state.error_message = error_display
            st.error(f"Error generating response: {str(e)}")
            with st.expander("Error Details", expanded=False):
                st.code(error_trace)
            logger.error(f"Error generating response: {str(e)}\n{error_trace}")
            st.session_state.conversation.append({
                "role": "bot", 
                "text": "I apologize, but I encountered an error while processing your question. Please try again."
            })
            st.session_state.processing = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload PDF files to get started")

