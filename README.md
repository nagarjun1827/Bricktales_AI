# QuoCo AI BOQ Processing System

A comprehensive AI-powered Bill of Quantities (BOQ) processing system with semantic search, price estimation, and intelligent data extraction capabilities.

## 🚀 Features

### Core Capabilities
- **Store BOQ Processing**: Extract and process complete BOQ files with rates, quantities, and amounts
- **TBE BOQ Processing**: Handle To-Be-Estimated BOQs (quantities only, no pricing) for future rate application
- **Semantic Search**: Find similar BOQ items using AI-powered vector embeddings
- **Price Fetching**: Get intelligent price recommendations based on historical data
- **Multi-Agent Architecture**: Leverages specialized agents for project extraction, location parsing, and item extraction

### AI-Powered Features
- **Gemini AI Integration**: Uses Google's Gemini 2.5 Flash for intelligent data extraction
- **Vector Embeddings**: Semantic similarity search using pgvector and text-embedding-004 model
- **Intelligent Column Detection**: Automatically identifies BOQ columns regardless of format
- **Pattern Matching**: Robust pattern recognition for item codes, units, and descriptions

## 📋 Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Features Deep Dive](#features-deep-dive)

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- PostgreSQL 14+ with pgvector extension
- Google Gemini API key

### Step 1: Install uv (Fast Python Package Manager)

`uv` is an extremely fast Python package installer and resolver, written in Rust. It's significantly faster than pip.

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

#### Verify Installation:
```bash
uv --version
```

### Step 2: Clone the Repository
```bash
git clone <repository-url>
cd bricktales-ai-boq
```

### Step 3: Create Virtual Environment with uv

#### Option 1: Using uv (Recommended - Fast!)
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

#### Option 2: Using Python's venv
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Step 4: Install Dependencies with uv

#### Using uv (Blazingly Fast! ⚡)
```bash
# Install all dependencies from pyproject.toml
uv pip install -e .

# Or install specific packages
uv pip install fastapi uvicorn psycopg2-binary pandas openpyxl \
    python-dotenv google-generativeai langchain langchain-google-genai \
    pydantic python-multipart
```

#### Using regular pip
```bash
pip install -e .
```

### Performance Comparison
```
 Installation Speed Comparison:
   uv:  ~10-20 seconds  ⚡⚡⚡
   pip: ~2-5 minutes    🐌
   
   uv is 10-100x faster than pip!
```

### Step 5: Verify Installation
```bash
# Check installed packages
uv pip list

# Or with pip
pip list
```

## ⚙️ Configuration

### Step 1: Create `.env` File
Create a `.env` file in the project root:

```env
# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=boq_database

# Upload Directory
UPLOAD_DIR=./uploads
```

### Step 2: Get Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## 🗄️ Database Setup

### Step 1: Install PostgreSQL
Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)

### Step 2: Install pgvector Extension
```sql
-- Connect to your database
psql -U postgres -d boq_database

-- Install pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 3: Create Database Schema
Run the following SQL to create the required tables:

```sql
-- Projects Table
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(500) NOT NULL,
    project_code VARCHAR(100) UNIQUE NOT NULL,
    client_id INTEGER,
    client_name VARCHAR(255),
    start_date DATE,
    end_date DATE,
    version INTEGER DEFAULT 1,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Locations Table
CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(project_id),
    location_name VARCHAR(255) NOT NULL,
    address TEXT,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Store BOQ Files (with rates)
CREATE TABLE store_boq_files (
    boq_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(project_id),
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Store BOQ Items (with rates and embeddings)
CREATE TABLE store_boq_items (
    item_id SERIAL PRIMARY KEY,
    boq_id INTEGER REFERENCES store_boq_files(boq_id),
    item_code VARCHAR(100),
    item_description TEXT NOT NULL,
    unit_of_measurement VARCHAR(50),
    quantity DECIMAL(15, 4),
    supply_unit_rate DECIMAL(15, 2),
    supply_amount DECIMAL(20, 2) GENERATED ALWAYS AS (quantity * supply_unit_rate) STORED,
    labour_unit_rate DECIMAL(15, 2),
    labour_amount DECIMAL(20, 2) GENERATED ALWAYS AS (quantity * labour_unit_rate) STORED,
    total_amount DECIMAL(20, 2) GENERATED ALWAYS AS (quantity * (supply_unit_rate + COALESCE(labour_unit_rate, 0))) STORED,
    location_id INTEGER REFERENCES locations(location_id),
    description_embedding vector(768),
    embedding_generated_at TIMESTAMP,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TBE BOQ Files (without rates)
CREATE TABLE to_be_estimated_boq_files (
    boq_id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(project_id),
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TBE BOQ Items (quantities only, no rates)
CREATE TABLE to_be_estimated_boq_items (
    item_id SERIAL PRIMARY KEY,
    boq_id INTEGER REFERENCES to_be_estimated_boq_files(boq_id),
    item_code VARCHAR(100),
    item_description TEXT NOT NULL,
    unit_of_measurement VARCHAR(50),
    quantity DECIMAL(15, 4),
    location_id INTEGER REFERENCES locations(location_id),
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_store_items_boq ON store_boq_items(boq_id);
CREATE INDEX idx_store_items_embedding ON store_boq_items USING ivfflat (description_embedding vector_cosine_ops);
CREATE INDEX idx_tbe_items_boq ON to_be_estimated_boq_items(boq_id);
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

#### 1. Process Store BOQ (with rates)
```bash
curl -X POST "http://localhost:8000/boq/upload" \
  -F "file=@your_boq_file.xlsx" \
  -F "uploaded_by=user123"
```

#### 2. Generate Embeddings
```bash
# Initialize database (first time only)
curl -X POST "http://localhost:8000/embeddings/initialize"

# Generate embeddings for a BOQ
curl -X POST "http://localhost:8000/embeddings/generate" \
  -H "Content-Type: application/json" \
  -d '{"boq_id": 1}'
```

#### 3. Fetch Prices for TBE BOQ
```bash
curl -X POST "http://localhost:8000/prices/fetch" \
  -H "Content-Type: application/json" \
  -d '{"boq_id": 1, "top_k": 5, "min_similarity": 0.5}'
```

#### 4. Search Similar Items
```bash
curl -X POST "http://localhost:8000/embeddings/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "excavation work", "top_k": 10}'
```

## 📚 API Endpoints

### Store BOQ Processing (`/boq`)
- `POST /boq/upload` - Upload and process BOQ file with rates
- `GET /boq/status/{task_id}` - Check processing status
- `GET /boq/result/{task_id}` - Get processing results

### TBE BOQ Processing (`/estimate`)
- `POST /estimate/upload` - Upload BOQ without rates
- `GET /estimate/status/{task_id}` - Check processing status
- `GET /estimate/result/{task_id}` - Get processing results
- `GET /estimate/items/{boq_id}` - Retrieve TBE items
- `GET /estimate/summary/{boq_id}` - Get BOQ summary

### Embeddings (`/embeddings`)
- `POST /embeddings/initialize` - Initialize vector database
- `POST /embeddings/generate` - Generate embeddings for BOQ
- `POST /embeddings/regenerate/{boq_id}` - Regenerate all embeddings
- `POST /embeddings/search` - Search similar items
- `GET /embeddings/status/{task_id}` - Check task status
- `GET /embeddings/result/{task_id}` - Get task results
- `GET /embeddings/stats` - Get embedding statistics

### Price Fetching (`/prices`)
- `POST /prices/fetch` - Fetch prices for BOQ items
- `GET /prices/status/{task_id}` - Check fetching status
- `GET /prices/result/{task_id}` - Get price recommendations
- `GET /prices/export/{task_id}/csv` - Export results to CSV
- `GET /prices/recommendations/{boq_id}` - Get recommendations directly

## 📁 Project Structure

```
bricktales-ai-boq/
├── app/
│   ├── agents/              # AI agents for extraction
│   │   ├── base_agent.py
│   │   ├── langchain_tools.py
│   │   ├── structure_agent.py
│   │   ├── project_agent.py
│   │   ├── location_agent.py
│   │   └── item_agent.py
│   ├── api/
│   │   └── v1/
│   │       ├── boq_router.py
│   │       ├── tbe_router.py
│   │       ├── embedding_router.py
│   │       └── price_router.py
│   ├── core/
│   │   └── config.py         # Configuration management
│   ├── models/
│   │   ├── domain.py          # Domain models
│   │   ├── dto.py             # Data transfer objects
│   │   ├── embedding_domain.py
│   │   ├── embedding_dto.py
│   │   ├── tbe_domain.py
│   │   ├── tbe_dto.py
│   │   └── price_dto.py
│   ├── repositories/
│   │   ├── boq_repository.py
│   │   ├── tbe_repository.py
│   │   ├── embedding_repository.py
│   │   └── price_repository.py
│   ├── services/
│   │   ├── boq_service.py
│   │   ├── tbe_service.py
│   │   ├── tbe_pattern_matcher.py
│   │   ├── embedding_service.py
│   │   └── price_service.py
│   └── tasks/
│       └── background.py      # Background task management
├── uploads/                   # File upload directory
├── .env                       # Environment configuration
├── .gitignore
├── .python-version           # Python version specification
├── main.py                    # FastAPI application entry
├── pyproject.toml            # Project dependencies
└── README.md
```

##  Features Deep Dive

### Multi-Agent Architecture
The system uses specialized AI agents:
- **Project Extractor**: Identifies project name, code, client, and dates
- **Location Extractor**: Extracts location and address information
- **Structure Analyzer**: Analyzes sheet structure and column layout
- **Item Extractor**: Intelligently extracts BOQ line items

### Semantic Search
- Uses Google's text-embedding-004 model (768 dimensions)
- Stores embeddings in PostgreSQL with pgvector
- Supports cosine similarity search
- Filters by BOQ, location, and similarity threshold

### Price Fetching
- Generates embeddings for TBE items on-the-fly
- Finds similar items from historical store BOQs
- Calculates price statistics (avg, min, max, median)
- Considers unit of measurement matching
- Provides detailed similar item information

### Pattern Matching
- Intelligent column detection (item code, description, quantity, unit, rate)
- Unit normalization (sqm, cum, mtr, kg, nos, etc.)
- Numeric value extraction from mixed content
- Valid item code detection

## 🔧 Troubleshooting

### Common Issues

**Issue**: `pgvector extension not found`
```sql
-- Solution: Install pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

**Issue**: `GEMINI_API_KEY not set`
```bash
# Solution: Add to .env file
GEMINI_API_KEY=your_key_here
```

**Issue**: Database connection error
```bash
# Solution: Verify PostgreSQL is running
# On Linux/macOS
sudo service postgresql status

# On Windows
# Check Services app for PostgreSQL service

# Check .env credentials match your PostgreSQL setup
```

**Issue**: uv command not found after installation
```bash
# Solution: Restart your terminal or add to PATH manually
# On macOS/Linux, add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.cargo/bin:$PATH"

# On Windows, the installer should handle PATH automatically
# If not, add to system PATH: %USERPROFILE%\.cargo\bin
```

**Issue**: Slow package installation
```bash
# Solution: Use uv instead of pip!
uv pip install <package>

# uv is 10-100x faster than pip
```

## 💡 uv Pro Tips

### Speed Up Development
```bash
# Install dependencies in parallel (uv does this by default!)
uv pip install -r requirements.txt

# Sync environment with pyproject.toml
uv pip sync

# Compile requirements for reproducible builds
uv pip compile pyproject.toml -o requirements.txt

# Create temporary virtual environment for testing
uv venv --seed temp-env
```

### Why use uv?
- ⚡ **10-100x faster** than pip
- 🔒 **Drop-in replacement** for pip
- 🎯 **Better dependency resolution**
- 📦 **Smaller disk usage**
- 🦀 **Written in Rust** for maximum performance

## 📝 Example Usage

### Processing a Store BOQ
```python
import requests

# Upload file
files = {'file': open('boq.xlsx', 'rb')}
data = {'uploaded_by': 'john_doe'}
response = requests.post('http://localhost:8000/boq/upload', files=files, data=data)
task_id = response.json()['task_id']

# Check status
status = requests.get(f'http://localhost:8000/boq/status/{task_id}')
print(status.json())

# Get results when complete
result = requests.get(f'http://localhost:8000/boq/result/{task_id}')
print(result.json())
```

### Searching for Similar Items
```python
search_payload = {
    "query": "excavation in hard rock",
    "top_k": 10,
    "min_similarity": 0.7
}
response = requests.post(
    'http://localhost:8000/embeddings/search',
    json=search_payload
)
similar_items = response.json()['items']
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.

## 🙏 Acknowledgments

- **uv** by Astral for blazingly fast package management
- **Google Gemini AI** for intelligent text extraction
- **LangChain** for agent orchestration
- **pgvector** for vector similarity search
- **FastAPI** for the robust API framework

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact the development team

---

**Version**: 2.0.0  
**Last Updated**: 2025  
**Powered by**: uv ⚡
