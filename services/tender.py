# services/tender.py
"""
Tender document processing service with ingestion, retrieval, and QA capabilities.
"""
import json
import random
import time
import concurrent.futures
import re
import requests
import PyPDF2
import logging
from io import BytesIO
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi
from datetime import datetime

import google.generativeai as genai
from core.settings import settings
from repositories.tender import TenderRepository

logger = logging.getLogger(__name__)

# Configuration
MAX_CHUNK_SIZE = 1000
HYBRID_SEARCH_ALPHA = 0.7
TOP_K_CHUNKS = 5
MAX_RETRIES = 3
MAX_PARALLEL_WORKERS = 10

genai.configure(api_key=settings.GEMINI_API_KEY)
embedding_model = "models/text-embedding-004"

summary_agent_model = genai.GenerativeModel(
    'models/gemini-2.5-flash',
    generation_config={"temperature": 0.7, "top_p": 0.95}
)

qa_agent_model = genai.GenerativeModel(
    'models/gemini-2.5-flash',
    generation_config={"temperature": 0.4, "top_p": 0.9}
)


class TenderService:
    """Unified service for tender document processing."""
    
    def __init__(self):
        self.tender_repo = TenderRepository()
    
    # ========== INGESTION METHODS ==========
    
    async def process_tender_document(self, file_url: str, uploaded_by: str = "user") -> Dict:
        """Main ingestion pipeline for tender documents."""
        start_time = time.time()
        
        try:
            logger.info("="*70)
            logger.info("🚀 PROCESSING TENDER DOCUMENT")
            logger.info("="*70)
            
            # Step 1: Fetch PDF
            raw_text = self._fetch_pdf_text(file_url)
            if not raw_text:
                return {"error": "Failed to fetch PDF", "tender_file_id": -1}
            
            # Step 2: Extract structured data
            logger.info("🤖 Extracting structured data...")
            structured_data = self._extract_json_from_text(raw_text)
            
            # Step 3: Chunk document
            logger.info("Chunking document...")
            chunks = self._chunk_text(raw_text, MAX_CHUNK_SIZE)
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings in parallel
            logger.info("⚡ Generating hybrid embeddings (in parallel)...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Dense embeddings
            logger.info(f"   → Dense embeddings using {MAX_PARALLEL_WORKERS} workers...")
            dense_embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                dense_embeddings = list(executor.map(
                    lambda text: self._safe_embed_content(text, "retrieval_document"),
                    chunk_texts
                ))
            
            # Sparse embeddings
            logger.info(f"   → Sparse embeddings using {MAX_PARALLEL_WORKERS} workers...")
            tokenized_chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                tokenized_chunks = list(executor.map(self._preprocess_text, chunk_texts))
            
            sparse_embeddings = []
            bm25_corpus = {}
            
            if tokenized_chunks and not all(len(t) == 0 for t in tokenized_chunks):
                bm25 = BM25Okapi(tokenized_chunks)
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                    sparse_embeddings = list(executor.map(self._create_sparse_vector, tokenized_chunks))
                
                bm25_corpus = {
                    'avg_doc_len': bm25.avgdl,
                    'doc_lens': bm25.doc_len,
                }
            else:
                sparse_embeddings = [{} for _ in tokenized_chunks]
            
            logger.info(f"✓ Generated {len(dense_embeddings)} hybrid embeddings")
            
            # Step 5: Store in database
            logger.info("Storing in database...")
            
            # Create project
            project_id = random.randint(10000, 99999)
            tender_number = f"TENDER-{project_id}"
            
            project_id = await self.tender_repo.create_project(
                project_name=structured_data.get('file_name', 'Untitled Tender'),
                project_code=f"TENDER-{datetime.now():%Y%m%d%H%M}",
                created_by=uploaded_by
            )
            
            # Create tender project
            tender_id = await self.tender_repo.create_tender_project(
                project_id=project_id,
                tender_number=tender_number,
                tender_date=structured_data.get('tender_date'),
                submission_deadline=structured_data.get('submission_deadline'),
                tender_status=structured_data.get('tender_status', 'Open'),
                created_by=uploaded_by
            )
            
            # Create tender file
            tender_file_id = await self.tender_repo.create_tender_file(
                tender_id=tender_id,
                file_name=structured_data.get('file_name', 'Untitled Document'),
                file_path=file_url,
                file_type='pdf',
                bm25_corpus=bm25_corpus,
                created_by=uploaded_by
            )
            
            # Prepare chunk data for bulk insert
            chunk_data = []
            for chunk, dense_emb, sparse_emb, tokens in zip(chunks, dense_embeddings, sparse_embeddings, tokenized_chunks):
                chunk_data.append({
                    'tender_file_id': tender_file_id,
                    'chunk_index': chunk['chunk_index'],
                    'chunk_text': chunk['text'],
                    'chunk_metadata': chunk['metadata'],
                    'dense_embedding': dense_emb,
                    'sparse_embedding': sparse_emb,
                    'bm25_tokens': tokens
                })
            
            # Bulk insert chunks
            rows_inserted = await self.tender_repo.bulk_insert_chunks(chunk_data)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✓ Stored Project (ID: {project_id}), Tender (ID: {tender_id}), File (ID: {tender_file_id}), and {rows_inserted} chunks")
            logger.info(f"\n✓ Document processed successfully in {processing_time:.2f}s!")
            
            return {
                "tender_file_id": tender_file_id,
                "tender_id": tender_id,
                "project_id": project_id,
                "chunks_created": rows_inserted,
                "processing_time": processing_time,
                "message": "Document processed successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in ingestion: {e}", exc_info=True)
            return {
                "error": str(e), 
                "tender_file_id": -1,
                "tender_id": -1,
                "project_id": -1,
                "chunks_created": 0,
                "processing_time": 0.0,
                "message": f"Error: {str(e)}"
            }
    
    # ========== RETRIEVAL METHODS ==========
    
    async def retrieve_chunks(self, tender_file_id: int, query: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant chunks using hybrid search."""
        try:
            # If no query, return first 5 chunks
            if not query or not query.strip():
                chunks = await self.tender_repo.get_chunks_by_file_id(tender_file_id, limit=5)
                return [
                    {'chunk_index': chunk['chunk_index'], 'text': chunk['chunk_text'], 'score': 1.0}
                    for chunk in chunks
                ]
            
            # Generate query embeddings
            query_dense = self._safe_embed_content(query, "retrieval_query")
            query_tokens = self._preprocess_text(query)
            query_sparse = self._create_sparse_vector(query_tokens)
            
            # Dense search
            dense_results = await self.tender_repo.search_chunks_dense(tender_file_id, query_dense, limit=10)
            
            # Sparse search
            all_chunks = await self.tender_repo.get_all_chunks_with_sparse(tender_file_id)
            
            sparse_scores = {}
            for chunk in all_chunks:
                if chunk['sparse_embedding']:
                    score = sum(
                        chunk['sparse_embedding'].get(str(k), 0) * v 
                        for k, v in query_sparse.items()
                    )
                    sparse_scores[chunk['chunk_index']] = score
            
            # Combine scores
            alpha = HYBRID_SEARCH_ALPHA
            combined_scores = {}
            
            for result in dense_results:
                chunk_idx = result['chunk_index']
                dense_score = result['dense_score']
                sparse_score = sparse_scores.get(chunk_idx, 0)
                combined_scores[chunk_idx] = alpha * dense_score + (1 - alpha) * sparse_score
            
            # Sort and get top K
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_CHUNKS]
            
            # Fetch chunk texts
            retrieved_chunks = []
            for chunk_idx, score in sorted_results:
                text = await self.tender_repo.get_chunk_by_index(tender_file_id, chunk_idx)
                if text:
                    retrieved_chunks.append({
                        'chunk_index': chunk_idx,
                        'text': text,
                        'score': score
                    })
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks: {e}", exc_info=True)
            return []
    
    # ========== AGENT METHODS ==========
    
    async def get_tender_file_details(self, tender_file_id: int) -> Optional[Dict]:
        """Get tender file details."""
        return await self.tender_repo.get_tender_file_by_id(tender_file_id)
    
    async def _validate_tender_file(self, tender_file_id: int) -> bool:
        """Validate if tender file exists."""
        return await self.tender_repo.check_file_exists(tender_file_id)
    
    async def generate_summary(self, tender_file_id: int, explanation_level: str = "professional") -> str:
        """Generate document summary."""
        try:
            # Validate file exists
            if not await self._validate_tender_file(tender_file_id):
                return "Error: Tender file not found"
            
            logger.info("\n" + "="*70)
            logger.info(f"📝 GENERATING SUMMARY (Mode: {explanation_level})")
            logger.info("="*70)
            
            # Get chunks
            chunks = await self.tender_repo.get_chunks_by_file_id(tender_file_id, limit=10)
            if not chunks:
                return "I could not generate a summary as no text was retrieved from the document."
            
            document_text = "\n\n".join([chunk['chunk_text'] for chunk in chunks])
            
            # Generate prompt
            if explanation_level == 'simple':
                prompt = f"""
                You are a friendly teacher explaining a tender document to a 14-year-old student. 
                Make it super easy to understand! Use simple words, short sentences, and emojis.
                
                Document content:
                {document_text}
                
                Please create a summary that covers:
                - 📋 What is this tender about?
                - 🎯 Who can apply?
                - 📅 Important dates?
                - 📝 What do they need to submit?
                """
            else:
                prompt = f"""
                Analyze this tender document and provide a comprehensive professional summary.
                
                Document content:
                {document_text}
                
                Provide a structured summary covering:
                1. Main purpose and scope
                2. Eligibility criteria
                3. Key requirements
                4. Important deadlines
                5. Submission details
                """
            
            response = summary_agent_model.generate_content(prompt)
            summary_text = response.text
            
            # Update in database
            is_simple = (explanation_level == 'simple')
            await self.tender_repo.update_tender_file_summary(tender_file_id, summary_text, is_simple)
            
            logger.info("✓ Summary generated")
            return summary_text
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def answer_question(self, tender_file_id: int, question: str, 
                             explanation_level: str = "professional") -> str:
        """Answer question about tender document."""
        try:
            # Validate file exists
            if not await self._validate_tender_file(tender_file_id):
                return "Error: Tender file not found"
            
            logger.info("\n" + "="*70)
            logger.info(f"❓ Q&A AGENT (Mode: {explanation_level})")
            logger.info(f"Question: {question}")
            logger.info("="*70)
            
            # Retrieve relevant chunks
            retrieved_chunks = await self.retrieve_chunks(tender_file_id, question)
            logger.info(f"-> Retrieved {len(retrieved_chunks)} relevant chunks")
            
            if not retrieved_chunks:
                return "I could not find an answer as no relevant text was retrieved from the document."
            
            context = "\n\n".join([
                f"[Chunk {chunk['chunk_index']}]:\n{chunk['text']}"
                for chunk in retrieved_chunks
            ])
            
            # Generate prompt
            if explanation_level == 'simple':
                prompt = f"""
                You are a friendly tutor helping a 14-year-old understand a tender document.
                Use simple language, short sentences, and emojis 😊.
                
                Here's what the document says:
                {context}
                
                Student's question: {question}
                
                Explain the answer in a way a 14-year-old can easily understand:
                """
            else:
                prompt = f"""
                You are an expert assistant analyzing tender documents.
                Provide a clear, accurate answer based ONLY on the context.
                If the answer is not in the context, say "The answer is not found in the provided document sections."
                
                Context from the document:
                {context}
                
                Question: {question}
                
                Answer:
                """
            
            logger.info("-> Generating answer...")
            response = qa_agent_model.generate_content(prompt)
            
            logger.info("\n" + "="*70)
            logger.info("ANSWER:")
            logger.info("="*70)
            logger.info(response.text)
            logger.info("="*70)
            
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    # ========== HELPER METHODS ==========
    
    def _fetch_pdf_text(self, url: str) -> Optional[str]:
        """Download PDF from URL and extract text."""
        try:
            logger.info(f"Fetching PDF from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=30, headers=headers, verify=False)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            raw_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    raw_text += f"\n--- Page {page_num + 1} ---\n"
                    raw_text += page_text
            
            raw_text = re.sub(r'(\n\s*)+\n', '\n', raw_text)
            logger.info(f"✓ Extracted {len(raw_text)} characters")
            return raw_text
            
        except Exception as e:
            logger.error(f"❌ Error fetching PDF: {e}")
            return None
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """Uses Gemini to extract structured JSON from raw text."""
        prompt_text = text[:12000]

        prompt = f"""
        You are an expert data extractor. Analyze the following tender document text 
        and extract the specified fields.
        
        Respond ONLY with a single, valid JSON object.
        
        Fields to extract:
        - "tender_date": The main date of the tender notice (YYYY-MM-DDTHH:MM:SS). If not found, use null.
        - "submission_deadline": The final date and time for submission (YYYY-MM-DDTHH:MM:SS). If not found, use null.
        - "tender_status": The current status. Default to "Open".
        - "file_name": A good, short name for this document, e.g., "Main Tender Document".
        
        Text:
        "{prompt_text}"
        
        JSON:
        """
        
        try:
            response = qa_agent_model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match:
                return {}
                
            json_data = json.loads(json_match.group(0))
            
            # Parse dates
            if 'tender_date' in json_data and json_data['tender_date']:
                try:
                    json_data['tender_date'] = datetime.fromisoformat(
                        str(json_data['tender_date']).replace('Z', '+00:00')
                    )
                except:
                    json_data['tender_date'] = None
            
            if 'submission_deadline' in json_data and json_data['submission_deadline']:
                try:
                    deadline_str = str(json_data['submission_deadline']).replace('Z', '+00:00')
                    if '+' not in deadline_str:
                        deadline_str += '+00:00'
                    json_data['submission_deadline'] = datetime.fromisoformat(deadline_str)
                except:
                    json_data['submission_deadline'] = None
            
            return json_data
            
        except Exception as e:
            logger.warning(f"   ⚠️ Error extracting JSON: {e}")
            return {}
    
    def _chunk_text(self, text: str, max_chunk_size: int) -> List[Dict]:
        """Chunk text into manageable pieces."""
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
    
    def _safe_embed_content(self, text: str, task_type: str, max_retries: int = MAX_RETRIES) -> List[float]:
        """Call embedding API with retry logic."""
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
                    logger.warning(f"   ⚠️ Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"   ⚠️ Embedding error: {e}")
                    return [0.0] * 768
        
        return [0.0] * 768
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25."""
        if not text:
            return []
        text = text.lower()
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        
        stopwords = {'that', 'this', 'with', 'from', 'have', 'will', 'your', 'their', 
                    'which', 'were', 'been', 'there', 'would', 'about', 'should', 'could'}
        
        return [token for token in tokens if token not in stopwords and len(token) > 2]
    
    def _create_sparse_vector(self, tokens: List[str], vocab_size: int = 10000) -> Dict[str, int]:
        """Create sparse vector representation from tokens."""
        from collections import Counter
        token_counts = Counter(tokens)
        sparse_vector = {}
        for token, count in token_counts.items():
            token_hash = hash(token) % vocab_size
            sparse_vector[str(token_hash)] = count
        return sparse_vector