# utils/embedding_utils.py
import time
import google.generativeai as genai
from typing import List
from config import embedding_model, MAX_RETRIES

def safe_embed_content(text: str, task_type: str = "retrieval_document", max_retries: int = MAX_RETRIES) -> List[float]:
    """Call embedding API with retry logic"""
    for attempt in range(max_retries):
        try:
            if not text or not text.strip():
                return [0.0] * 768
            
            result = genai.embed_content(
                model=embedding_model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = (attempt + 1) * 5
                print(f"   ⚠️ Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"   ⚠️ Embedding error: {e}")
                return [0.0] * 768
    
    return [0.0] * 768

def batch_embed_content(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Batch embed multiple texts"""
    embeddings = []
    for text in texts:
        embedding = safe_embed_content(text, task_type)
        embeddings.append(embedding)
    return embeddings