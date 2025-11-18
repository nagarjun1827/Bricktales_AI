# repositories/tender.py
"""
Async Repository for tender operations using SQLAlchemy ORM.
"""
import logging
from typing import List, Dict, Optional
from sqlalchemy import select, delete, func, literal_column
from sqlalchemy.ext.asyncio import AsyncSession
from connections.postgres_connection import DatabaseConnection
from models.tender_models import TenderProject, TenderFile, TenderChunk
from models.project_models import Project

logger = logging.getLogger(__name__)


class TenderRepository:
    """Async repository for tender operations."""
    
    def __init__(self):
        self.AsyncSessionLocal = DatabaseConnection.get_async_session_factory()
    
    async def create_tender_project(self, project_id: int, tender_number: str, 
                                   tender_date=None, submission_deadline=None,
                                   tender_status: str = "Open", created_by: str = "system") -> int:
        """Create a new tender project."""
        async with self.AsyncSessionLocal() as session:
            tender_project = TenderProject(
                project_id=project_id,
                tender_number=tender_number,
                tender_date=tender_date,
                submission_deadline=submission_deadline,
                tender_status=tender_status,
                created_by=created_by
            )
            session.add(tender_project)
            await session.commit()
            await session.refresh(tender_project)
            return tender_project.tender_id
    
    async def create_tender_file(self, tender_id: int, file_name: str, file_path: str,
                                file_type: str = "pdf", bm25_corpus: Optional[Dict] = None,
                                created_by: str = "system") -> int:
        """Create a new tender file."""
        async with self.AsyncSessionLocal() as session:
            tender_file = TenderFile(
                tender_id=tender_id,
                file_name=file_name,
                file_path=file_path,
                file_type=file_type,
                bm25_corpus=bm25_corpus,
                created_by=created_by
            )
            session.add(tender_file)
            await session.commit()
            await session.refresh(tender_file)
            return tender_file.tender_file_id
    
    async def bulk_insert_chunks(self, chunks: List[Dict]) -> int:
        """Bulk insert tender chunks."""
        async with self.AsyncSessionLocal() as session:
            chunk_objects = []
            for chunk in chunks:
                chunk_obj = TenderChunk(
                    tender_file_id=chunk['tender_file_id'],
                    chunk_index=chunk['chunk_index'],
                    chunk_text=chunk['chunk_text'],
                    chunk_metadata=chunk['chunk_metadata'],
                    dense_embedding=chunk['dense_embedding'],
                    sparse_embedding=chunk['sparse_embedding'],
                    bm25_tokens=chunk['bm25_tokens']
                )
                chunk_objects.append(chunk_obj)
            
            session.add_all(chunk_objects)
            await session.commit()
            return len(chunk_objects)
    
    async def get_tender_file_by_id(self, tender_file_id: int) -> Optional[Dict]:
        """Get tender file by ID."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(TenderFile).where(TenderFile.tender_file_id == tender_file_id)
            )
            tender_file = result.scalar_one_or_none()
            
            if tender_file:
                return {
                    'tender_file_id': tender_file.tender_file_id,
                    'tender_id': tender_file.tender_id,
                    'file_name': tender_file.file_name,
                    'file_path': tender_file.file_path,
                    'file_type': tender_file.file_type,
                    'summary': tender_file.summary,
                    'simple_summary': tender_file.simple_summary,
                    'created_at': tender_file.created_at.isoformat() if tender_file.created_at else None
                }
            return None
    
    async def get_chunks_by_file_id(self, tender_file_id: int, limit: Optional[int] = None) -> List[Dict]:
        """Get all chunks for a tender file."""
        async with self.AsyncSessionLocal() as session:
            query = select(TenderChunk).where(
                TenderChunk.tender_file_id == tender_file_id
            ).order_by(TenderChunk.chunk_index)
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            chunks = result.scalars().all()
            
            return [
                {
                    'chunk_index': chunk.chunk_index,
                    'chunk_text': chunk.chunk_text,
                    'chunk_metadata': chunk.chunk_metadata
                }
                for chunk in chunks
            ]
    
    async def search_chunks_dense(self, tender_file_id: int, query_embedding: List[float], 
                                 limit: int = 10) -> List[Dict]:
        """Search chunks using dense embeddings."""
        async with self.AsyncSessionLocal() as session:
            query_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            similarity = literal_column(
                f"1 - (dense_embedding <=> '{query_str}'::vector)"
            ).label('dense_score')
            
            query = select(
                TenderChunk.chunk_index,
                TenderChunk.chunk_text,
                similarity
            ).where(
                TenderChunk.tender_file_id == tender_file_id,
                TenderChunk.dense_embedding.isnot(None)
            ).order_by(
                literal_column(f"dense_embedding <=> '{query_str}'::vector")
            ).limit(limit)
            
            result = await session.execute(query)
            rows = result.all()
            
            return [
                {
                    'chunk_index': row[0],
                    'chunk_text': row[1],
                    'dense_score': float(row[2])
                }
                for row in rows
            ]
    
    async def get_all_chunks_with_sparse(self, tender_file_id: int) -> List[Dict]:
        """Get all chunks with sparse embeddings."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    TenderChunk.chunk_index,
                    TenderChunk.chunk_text,
                    TenderChunk.sparse_embedding
                ).where(TenderChunk.tender_file_id == tender_file_id)
            )
            rows = result.all()
            
            return [
                {
                    'chunk_index': row[0],
                    'chunk_text': row[1],
                    'sparse_embedding': row[2]
                }
                for row in rows
            ]
    
    async def get_chunk_by_index(self, tender_file_id: int, chunk_index: int) -> Optional[str]:
        """Get chunk text by index."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(TenderChunk.chunk_text).where(
                    TenderChunk.tender_file_id == tender_file_id,
                    TenderChunk.chunk_index == chunk_index
                )
            )
            chunk_text = result.scalar_one_or_none()
            return chunk_text
    
    async def update_tender_file_summary(self, tender_file_id: int, summary: str, 
                                        is_simple: bool = False):
        """Update tender file summary."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(TenderFile).where(TenderFile.tender_file_id == tender_file_id)
            )
            tender_file = result.scalar_one_or_none()
            
            if tender_file:
                if is_simple:
                    tender_file.simple_summary = summary
                else:
                    tender_file.summary = summary
                
                await session.commit()
    
    async def check_file_exists(self, tender_file_id: int) -> bool:
        """Check if tender file exists."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.count()).select_from(TenderFile).where(
                    TenderFile.tender_file_id == tender_file_id
                )
            )
            count = result.scalar()
            return count > 0
    
    async def create_project(self, project_name: str, project_code: str, 
                           created_by: str = "system") -> int:
        """Create a main project."""
        async with self.AsyncSessionLocal() as session:
            project = Project(
                project_name=project_name,
                project_code=project_code,
                project_type="tender",
                created_by=created_by
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)
            return project.project_id