# QuoCo - An AI powered BOQ Estimator and Tender Analyzer

An advanced AI-powered Bill of Quantities (BOQ) processing, tender document analysis, and estimation system built with FastAPI, PostgreSQL, and Google’s Gemini AI.

## 🚀 Features

### 1. Store BOQ (With Rates)
- Process BOQ files containing supply and labour rates
- Automatic rate extraction and validation
- Intelligent rate derivation when data is missing
- Generate semantic embeddings for similarity search
- Store pricing data for future estimates

### 2. Estimate BOQ (Without Rates)
- Process BOQ files without rates
- Automatic price estimation using vector similarity search
- Match items with historical pricing data
- Generate detailed Excel reports with:
  - Summary sheet with totals and statistics
  - Detailed items with pricing sources
  - Similarity scores for transparency
- Export estimated BOQs with confidence metrics

### 3. Tender Document Processing
- Ingest PDF tender documents
- Extract structured metadata using AI
- Hybrid search
- Generate summaries at different complexity levels
- Q&A capabilities for document interrogation

## 🏗️ Architecture

### Three-Tier Architecture
```
┌─────────────────┐
│    Routers      │  FastAPI endpoints
├─────────────────┤
│    Services     │  Business logic
├─────────────────┤
│  Repositories   │  Database operations
└─────────────────┘
```

### Technology Stack
- **Framework**: FastAPI (async)
- **Database**: PostgreSQL with pgvector extension
- **ORM**: SQLAlchemy 2.0 (async)
- **AI**: Google Gemini (Flash 2.5)
- **Embeddings**: text-embedding-004 (768 dimensions)
- **File Processing**: pandas, openpyxl
- **PDF Processing**: PyPDF2, pdfplumber

## 📋 Prerequisites

- Python 3.12+
- PostgreSQL 14+ with pgvector extension
- Google Gemini API key

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd bricktales-ai-boq
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
# or using uv
uv sync
```

3. **Set up environment variables**
Create a `.env` file:
```env
GEMINI_API_KEY=<your_gemini_api_key>
DB_HOST=<your_db_host>
DB_PORT=<your_db_port>
DB_USER=<your_db_user>
DB_PASSWORD=<your_db_password>
DB_NAME=<your_db_boq_name>
```

4. **Initialize database**
```bash
python -c "from connections.db_init import init_db; init_db(create_tables=True)"
```

5. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

Once running, access the interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Core Workflows

### Store BOQ Workflow
```python
# Upload BOQ with rates
POST /store-boq/upload
{
    "file_url": "http://example.com/boq.xlsx",
    "uploaded_by": "user123"
}

# Track progress
GET /store-boq/status/{task_id}

# Get results
GET /store-boq/result/{task_id}
```

### Estimate BOQ Workflow
```python
# Upload BOQ without rates
POST /estimate-boq/upload
{
    "file_url": "http://example.com/estimate.xlsx",
    "uploaded_by": "user123",
    "min_similarity": 0.5,
    "export_excel": true
}

# Track progress
GET /estimate-boq/status/{task_id}

# Get results with prices
GET /estimate-boq/result/{task_id}

# Download Excel report
GET /estimate-boq/download-excel/{task_id}
```

### Tender Document Workflow
```python
# Ingest tender PDF
POST /tender/ingest
{
    "url": "http://example.com/tender.pdf",
    "uploaded_by": "user123"
}

# Generate summary
POST /tender/{tender_file_id}/summarize
{
    "explanation_level": "professional"  # or "simple"
}

# Ask questions
POST /tender/{tender_file_id}/query
{
    "question": "What are the eligibility criteria?",
    "explanation_level": "professional"
}
```

## 🗄️ Database Schema

### Core Tables
- **projects**: Main project information
- **store_boq_projects**: Store BOQ metadata
- **store_boq_files**: BOQ file records
- **store_boq_items**: Line items with rates (computed columns)
- **estimate_boq_projects**: Estimate BOQ metadata
- **estimate_boq_files**: TBE BOQ file records
- **estimate_boq_items**: Line items without rates
- **tender_projects**: Tender metadata
- **tender_files**: Tender document records
- **tender_chunks**: Document chunks with embeddings

### Key Features
- **pgvector**: Stores 768-dimensional embeddings
- **Computed Columns**: Auto-calculate amounts (supply, labour, total)
- **Foreign Keys**: Enforce referential integrity
- **Timestamps**: Track creation and updates

## 🎯 Key Features Deep Dive

### 1. Parallel Embedding Generation
- Batch processing (20-100 items per batch)
- Retry logic with exponential backoff
- 20-40x faster than sequential processing
- Handles rate limiting gracefully

### 2. Intelligent Rate Detection
- AI-powered column structure analysis
- Pattern matching fallback for rate columns
- Derives missing rates from amounts
- Validates data consistency

### 3. Vector Similarity Search
- Cosine similarity for semantic matching
- Filters by unit of measurement
- Configurable similarity threshold
- Returns pricing source and confidence score

### 4. Excel Export with Summary
- Professional formatting
- Summary sheet with statistics
- Detailed items with pricing
- Color-coded sections

### 5. Hybrid Search (Tender)
- Dense embeddings (semantic)
- Sparse embeddings (BM25)
- Weighted combination (α = 0.7)
- Top-K retrieval

## 📁 Project Structure
```
bricktales-ai-boq/
├── agents/                 # AI agents for data extraction
│   ├── gemini_tools.py    # Gemini-based extraction tools
│   └── item_extractor.py  # BOQ item extraction
├── connections/           # Database connections
│   ├── db_init.py        # DB initialization
│   └── postgres_connection.py  # Connection pooling
├── core/                  # Core configuration
│   └── settings.py       # Environment settings
├── dto/                   # Data transfer objects
│   ├── request_dto/      # API request schemas
│   └── response_dto/     # API response schemas
├── models/               # SQLAlchemy models
│   ├── base.py          # Base model
│   ├── project_models.py
│   ├── store_boq_models.py
│   ├── estimate_boq_models.py
│   └── tender_models.py
├── repositories/         # Database operations
│   ├── store_boq.py
│   ├── estimate_boq.py
│   ├── price.py
│   └── tender.py
├── routers/             # API endpoints
│   ├── store_boq.py
│   ├── estimate_boq.py
│   └── tender.py
├── services/            # Business logic
│   ├── store_boq.py
│   ├── estimate_boq.py
│   └── tender.py
├── tasks/               # Background tasks
│   └── background_tasks.py
├── main.py              # FastAPI application
└── pyproject.toml       # Dependencies
```

## 🧪 Testing
```bash
# Run health check
curl http://localhost:8000/health

# Test Store BOQ
curl -X POST http://localhost:8000/store-boq/upload \
  -H "Content-Type: application/json" \
  -d '{"file_url": "http://example.com/boq.xlsx", "uploaded_by": "test"}'

# Test Estimate BOQ
curl -X POST http://localhost:8000/estimate-boq/upload \
  -H "Content-Type: application/json" \
  -d '{"file_url": "http://example.com/estimate.xlsx", "uploaded_by": "test", "min_similarity": 0.5, "export_excel": true}'
```

## 🔍 Monitoring & Logging

- Comprehensive logging at INFO level
- Request tracking via task IDs
- Processing time metrics
- Error tracking with stack traces

## ⚙️ Configuration

Key settings in `core/settings.py`:
- `GEMINI_API_KEY`: Google Gemini API key
- `DB_*`: PostgreSQL connection settings
- Database connection pooling (10 connections, 20 overflow)
- Auto-reconnect on connection loss

## 🚦 Performance

### Optimizations
- Async database operations throughout
- Connection pooling (sync + async)
- Parallel embedding generation
- Batch database operations
- Computed columns in database
- Strategic indexing

### Benchmarks
- Embedding generation: 20-40x faster with parallel processing
- BOQ processing: ~2-5s per 100 items
- Database operations: <100ms for most queries

## 🛡️ Error Handling

- Retry logic for API calls (3 attempts)
- Exponential backoff for rate limits
- Transaction rollback on failures
- Detailed error messages
- Graceful degradation

## 📝 Best Practices

1. **Always use async/await** for database operations
2. **Use ORM queries** instead of raw SQL
3. **Handle None values** in database results
4. **Convert dates properly** for asyncpg
5. **Use computed columns** for calculated fields
6. **Batch operations** for better performance
7. **Log extensively** for debugging

## 🔄 Migration from Sync to Async

The codebase has been fully migrated to async operations:
- ✅ All repositories use async SQLAlchemy
- ✅ All services use async/await
- ✅ All routers use async endpoints
- ✅ Connection pooling for both sync and async
- ✅ Proper session management

## 🤝 Contributing

1. Follow async/await patterns
2. Use type hints
3. Add comprehensive logging
4. Write docstrings
5. Test thoroughly

## 📄 License

[Your License Here]

## 👥 Team

- **Developers**: Nagarjun and Anuj
- **Project**: QuoCo
- **Version**: 1.0.0

## 📧 Support

For issues and questions, please open an issue on the repository.

---

Built with ❤️ using FastAPI and Google Gemini AI