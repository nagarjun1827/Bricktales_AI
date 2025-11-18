# BrickTales AI BOQ Processing System

A comprehensive AI-powered Bill of Quantities (BOQ) processing system with semantic search, intelligent price estimation, and automated data extraction capabilities.

## 🚀 Features

### Core Capabilities
- **Storing the BOQ**: Extract and store BOQ files with complete pricing (supply rates, labour rates, quantities)
- **Estimating the Prices of BOQ Line items**: Process BOQs without pricing and automatically fetch estimated rates from historical data
- **Semantic Search**: Find similar BOQ items using AI-powered vector embeddings (pgvector)
- **Intelligent Price Fetching**: Get price recommendations based on similarity matching with historical BOQ data
- **Multi-Agent Architecture**: Specialized AI agents for project extraction, location parsing, and item extraction

### AI-Powered Features
- **Gemini AI Integration**: Uses Google's Gemini 2.5 Flash for intelligent data extraction
- **Vector Embeddings**: Semantic similarity search using pgvector and text-embedding-004 model (768 dimensions)
- **Intelligent Column Detection**: Automatically identifies BOQ columns regardless of Excel format
- **Pattern Matching**: Robust pattern recognition for item codes, units, rates, and descriptions
- **Parallel Batch Processing**: High-performance embedding generation with retry logic

## 📋 Table of Contents
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Performance](#performance)

## 🏗️ Architecture

### Technology Stack
- **Backend Framework**: FastAPI (async)
- **Database**: PostgreSQL 14+ with pgvector extension
- **ORM**: SQLAlchemy 2.0 (async/await support)
- **AI/ML**: Google Gemini 2.5 Flash, text-embedding-004
- **Data Processing**: Pandas, OpenPyXL
- **Agent Framework**: LangChain with custom tools
- **Python Version**: 3.13+

### Three-Tier Architecture
```
┌─────────────────────────────────────────┐
│          API Layer (Routers)            │
│  - store_boq.py: Store BOQ endpoints    │
│  - estimate_boq.py: Estimate endpoints  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Service Layer (Business Logic)    │
│  - StoreBOQProcessor: File processing   │
│  - TBEBOQProcessor: Price estimation    │
│  - AI Agents: Data extraction           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     Repository Layer (Data Access)      │
│  - StoreBOQRepository: Store operations │
│  - TBEBOQRepository: Estimate ops       │
│  - PriceRepository: Similarity search   │
└─────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- PostgreSQL 14+ with pgvector extension
- Google Gemini API key

### Step 1: Install uv (Recommended - Fast Python Package Manager)

`uv` is an extremely fast Python package installer written in Rust (10-100x faster than pip).

#### On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative: Using pip
```bash
pip install uv
```

### Step 2: Clone Repository
```bash
git clone <repository-url>
cd bricktales-ai-boq
```

### Step 3: Create Virtual Environment
```bash
# Using uv (recommended)
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
# Using uv (blazingly fast! ⚡)
uv pip install -e .

# Or using regular pip
pip install -e .
```

**Performance Note**: uv installs packages 10-100x faster than pip!

## ⚙️ Configuration

### Create `.env` File
```env
# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=boq_database
```

### Get Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy to your `.env` file

## 🗄️ Database Setup

### Step 1: Install PostgreSQL
Download from [postgresql.org](https://www.postgresql.org/download/)

### Step 2: Install pgvector Extension
```sql
-- Connect to your database
psql -U postgres -d boq_database

-- Install pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 3: Database Schema

The system uses the following main tables:

**Projects Hierarchy:**
```
projects (main project info)
  ├── store_boq_projects (historical BOQs with rates)
  │   ├── store_boq_locations
  │   ├── store_boq_files
  │   └── store_boq_items (with embeddings, rates, computed amounts)
  └── estimate_boq_projects (BOQs needing estimation)
      ├── estimate_boq_locations
      ├── estimate_boq_files
      └── estimate_boq_items (with embeddings, quantities only)
```

**Key Features:**
- `store_boq_items` has computed columns for amounts (supply_amount, labour_amount, total_amount)
- Both item tables support 768-dimensional vector embeddings for semantic search
- Foreign key relationships maintain data integrity
- Async-compatible with asyncpg driver

See full schema in documentation or create tables via:
```python
from connections.db_init import init_db_async
await init_db_async(create_tables=True)
```

## 🚀 Usage

### Start the Server
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Basic Workflow

#### 1. Store BOQ (Historical Data with Rates)
```bash
curl -X POST "http://localhost:8000/store-boq/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "http://example.com/path/to/boq.xlsx",
    "uploaded_by": "user123"
  }'
```

**What it does:**
- Downloads Excel from URL
- Extracts project info, location, and items
- Identifies columns using AI (item code, description, quantity, unit, supply rate, labour rate)
- Stores items with rates in database
- Automatically generates 768-dimensional embeddings for semantic search
- Uses parallel batch processing for fast embedding generation

#### 2. Estimate BOQ (New BOQ Needing Prices)
```bash
curl -X POST "http://localhost:8000/estimate-boq/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "http://example.com/path/to/new_boq.xlsx",
    "uploaded_by": "user123",
    "min_similarity": 0.5,
    "export_excel": true
  }'
```

**What it does:**
- Downloads and processes Excel file
- Extracts items (descriptions, quantities, units)
- Generates embeddings for each item
- Finds best matching historical item using vector similarity
- Applies rates from best match (top_k=1)
- Calculates estimated amounts
- Optionally generates Excel with summary and detailed pricing

#### 3. Check Processing Status
```bash
curl -X GET "http://localhost:8000/estimate-boq/status/{task_id}"
```

#### 4. Get Results
```bash
curl -X GET "http://localhost:8000/estimate-boq/result/{task_id}"
```

#### 5. Download Excel (if generated)
```bash
curl -X GET "http://localhost:8000/estimate-boq/download-excel/{task_id}" \
  --output estimated_boq.xlsx
```

## 📚 API Endpoints

### Store BOQ (`/store-boq`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Process BOQ with rates from URL |
| `/status/{task_id}` | GET | Check processing status |
| `/result/{task_id}` | GET | Get processing results |
| `/info/{boq_id}` | GET | Get BOQ information |
| `/delete/{boq_id}` | DELETE | Delete BOQ and related data |

### Estimate BOQ (`/estimate-boq`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Process BOQ and fetch prices from URL |
| `/status/{task_id}` | GET | Check processing status |
| `/result/{task_id}` | GET | Get detailed results with pricing |
| `/download-excel/{task_id}` | GET | Download Excel with estimates |
| `/info/{boq_id}` | GET | Get BOQ information |
| `/delete/{boq_id}` | DELETE | Delete BOQ and related data |

### Health Check
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Async health check |
| `/health/sync` | GET | Sync health check |

## 📁 Project Structure
```
bricktales-ai-boq/
├── agents/                      # AI agents for data extraction
│   ├── base_agent.py           # Base agent class
│   ├── gemini_tools.py         # Gemini AI tools (structure, project, location)
│   └── item_extractor.py       # Item extraction agent
├── connections/                 # Database connections
│   ├── postgres_connection.py  # Singleton connection manager (sync/async)
│   └── db_init.py              # Database initialization utilities
├── core/                        # Core configuration
│   └── settings.py             # Application settings
├── dto/                         # Data Transfer Objects
│   ├── request_dto/            # Request models
│   └── response_dto/           # Response models
├── models/                      # SQLAlchemy ORM models
│   ├── base.py                 # Base declarative class
│   ├── project_models.py       # Project and company models
│   ├── store_boq_models.py     # Store BOQ models (with computed columns)
│   ├── estimate_boq_models.py  # Estimate BOQ models
│   └── tender_models.py        # Tender models
├── repositories/                # Data access layer
│   ├── store_boq.py            # Store BOQ repository (async)
│   ├── estimate_boq.py         # Estimate BOQ repository (async)
│   └── price.py                # Price fetching repository (async)
├── routers/                     # API routers
│   ├── store_boq.py            # Store BOQ endpoints
│   └── estimate_boq.py         # Estimate BOQ endpoints
├── services/                    # Business logic layer
│   ├── store_boq.py            # Store BOQ
│   ├── estimate_boq.py         # Estimate BOQ
│   └── pattern_matcher.py      # Pattern matching utilities
├── tasks/                       # Background task management
│   └── background_tasks.py     # Task tracking
├── .env                         # Environment configuration
├── main.py                      # FastAPI application entry point
├── pyproject.toml              # Project dependencies
└── README.md                    # This file
```

## 🔄 Workflow

### Store BOQ Workflow
```
1. Upload Excel via URL
2. Download and parse Excel file
3. Extract project info using Gemini AI
4. Extract location info using Gemini AI
5. Analyze sheet structure using Gemini AI
6. Extract line items (description, qty, unit, rates)
7. Store in database with computed amounts
8. Generate embeddings in parallel batches (20-100 items/batch)
9. Store embeddings for similarity search
```

### Estimate BOQ Workflow
```
1. Upload Excel via URL
2. Process file (same as Store BOQ, steps 1-6)
3. Extract items (description, qty, unit only)
4. Store in estimate_boq_items table
5. Generate embeddings for all items (parallel batches)
6. For each item:
   a. Use embedding to find top 1 most similar stored item
   b. Check similarity score > min_similarity threshold
   c. If match found: use its supply_rate and labour_rate
   d. Calculate estimated amounts (qty × rate)
7. Generate Excel with summary and detailed items (optional)
8. Return complete results with pricing sources
```

### Semantic Search Process
```
Query Item Description
       ↓
Generate 768-dim Embedding (text-embedding-004)
       ↓
Vector Similarity Search (pgvector cosine similarity)
       ↓
Filter by:
  - Unit of Measurement match
  - Minimum similarity threshold (default 0.5)
  - Must have supply_rate > 0
       ↓
Return Top K Matches (default K=1 for estimation)
       ↓
Apply rates from best match
```

## ⚡ Performance

### Embedding Generation
- **Parallel Batch Processing**: Processes 20-100 items per batch
- **Retry Logic**: Exponential backoff for rate limiting (429 errors)
- **Performance Gain**: 20-40x faster than sequential processing
- **Example**: 1000 items processed in ~15-30 seconds

### Database Operations
- **Async SQLAlchemy**: Non-blocking database operations
- **Connection Pooling**: Efficient connection management
- **Computed Columns**: Database-side amount calculations
- **Batch Inserts**: Bulk insert operations for items

### API Response Times
- **File Download**: Depends on file size and network
- **Extraction**: 2-5 seconds per sheet (AI processing)
- **Embedding Generation**: ~0.5-1 second per batch of 20 items
- **Similarity Search**: <100ms per item (with pgvector indexes)

## 🔧 Key Features Explained

### 1. Computed Columns
Store BOQ items use PostgreSQL GENERATED STORED columns:
```sql
supply_amount = quantity * supply_unit_rate
labour_amount = quantity * labour_unit_rate
total_amount = quantity * (supply_unit_rate + labour_unit_rate)
```
Benefits: Automatic calculation, no sync issues, always accurate

### 2. Vector Similarity Search
- Uses pgvector extension for efficient similarity search
- 768-dimensional embeddings from text-embedding-004
- Cosine similarity metric for matching
- IVFFlat index for fast searches on large datasets

### 3. Multi-Agent Architecture
- **Structure Agent**: Analyzes Excel layout, identifies columns
- **Project Agent**: Extracts project name, code, client, dates
- **Location Agent**: Extracts location and address info
- **Item Extractor**: Extracts line items with fallback pattern matching

### 4. Intelligent Column Detection
AI identifies columns even if headers are non-standard:
- "S.No" / "Item No" → item_code
- "Description" / "Particulars" / "Scope of Work" → description
- "Qty" / "Quantity" → quantity
- "UOM" / "Unit" → unit
- "Rate" / "Supply Rate" / "Material Rate" → supply_rate
- "Labour Rate" / "Labor Rate" → labour_rate

### 5. Rate Derivation Logic
If rates missing but amounts present:
```python
if supply_amount > 0 and quantity > 0:
    supply_rate = supply_amount / quantity
if labour_amount > 0 and quantity > 0:
    labour_rate = labour_amount / quantity
```

## 🔒 Error Handling

### Retry Mechanisms
- **Embedding Generation**: 3 retries with exponential backoff
- **Network Errors**: Handles timeouts, 500 errors, rate limits
- **Database Errors**: Transaction rollback on failures

### Data Validation
- Minimum row/column requirements for sheets
- Required columns validation (description, quantity)
- Unit normalization (Sqm, Cum, Mtr, etc.)
- Numeric value extraction with regex

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Best Practices

### When to Use Store BOQ
- You have historical BOQs with complete pricing
- Want to build a database of rates for future estimation
- Need to track project costs over time

### When to Use Estimate BOQ
- You have a new BOQ without pricing
- Want to estimate costs based on similar historical work
- Need quick price recommendations with source attribution

### Similarity Threshold Guidelines
- **0.7-1.0**: Very similar items (high confidence)
- **0.5-0.7**: Moderately similar (good for general estimates)
- **0.3-0.5**: Loosely similar (use with caution)
- **<0.3**: Not recommended for pricing

## 📄 License

This project is proprietary software. All rights reserved.

## 🙏 Acknowledgments

- **uv** by Astral for blazingly fast package management ⚡
- **Google Gemini AI** for intelligent text extraction
- **LangChain** for agent orchestration
- **pgvector** for vector similarity search
- **FastAPI** for the robust async API framework
- **SQLAlchemy** for powerful async ORM capabilities

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact the development team at support@bricktales.ai

---

**Version**: 2.0.0  
**Last Updated**: November 2025  
**Powered by**: Google Gemini AI, pgvector, FastAPI, SQLAlchemy, uv ⚡