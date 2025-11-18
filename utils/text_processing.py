# utils/text_processing.py
import re
import string
from collections import Counter
from typing import List, Dict

STOPWORDS = {
    'that', 'this', 'with', 'from', 'have', 'will', 'your', 'their', 'which',
    'were', 'been', 'there', 'would', 'about', 'should', 'could', 'these',
    'those', 'shall', 'must', 'and', 'the', 'for', 'is', 'in', 'it', 'to', 'of'
}

def preprocess_text(text: str) -> List[str]:
    """Tokenize and preprocess text for BM25"""
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]

def create_sparse_vector(tokens: List[str], vocab_size: int = 10000) -> Dict[str, int]:
    """Create sparse vector representation from tokens"""
    token_counts = Counter(tokens)
    sparse_vector = {}
    for token, count in token_counts.items():
        token_hash = hash(token) % vocab_size
        sparse_vector[str(token_hash)] = count
    return sparse_vector

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[Dict]:
    """Chunk text into manageable pieces"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) + 1 > max_chunk_size and current_chunk:
            chunks.append({
                "chunk_index": chunk_index,
                "text": current_chunk.strip(),
                "metadata": {"chunk_size": len(current_chunk), "chunk_type": "paragraph_group"}
            })
            current_chunk = ""
            chunk_index += 1
        
        if len(para) > max_chunk_size:
            if current_chunk:
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": current_chunk.strip(),
                    "metadata": {"chunk_size": len(current_chunk), "chunk_type": "paragraph_group"}
                })
                chunk_index += 1
                current_chunk = ""
            
            words = para.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                    temp_chunk += word + " "
                else:
                    if temp_chunk:
                        chunks.append({
                            "chunk_index": chunk_index,
                            "text": temp_chunk.strip(),
                            "metadata": {"chunk_size": len(temp_chunk), "chunk_type": "large_paragraph_split"}
                        })
                        chunk_index += 1
                    temp_chunk = word + " "
            
            if temp_chunk.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": temp_chunk.strip(),
                    "metadata": {"chunk_size": len(temp_chunk), "chunk_type": "large_paragraph_split"}
                })
                chunk_index += 1
        else:
            current_chunk += para + "\n\n"
    
    if current_chunk.strip():
        chunks.append({
            "chunk_index": chunk_index,
            "text": current_chunk.strip(),
            "metadata": {"chunk_size": len(current_chunk), "chunk_type": "final_chunk"}
        })
    
    return chunks