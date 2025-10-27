-- =====================================================
-- PROJECTS TABLE
-- =====================================================
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT,
    project_code VARCHAR(50) UNIQUE,
    client_name TEXT,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- LOCATIONS TABLE
-- =====================================================
CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    location_name TEXT,
    address TEXT,
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- BOQ_FILES TABLE
-- =====================================================
CREATE TABLE boq_files (
    boq_id SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT,
    file_type VARCHAR(20),
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    uploaded_by TEXT,
    version INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT
);

-- =====================================================
-- BOQ_ITEMS TABLE
-- =====================================================
CREATE TABLE boq_items (
    item_id SERIAL PRIMARY KEY,
    boq_id INT NOT NULL REFERENCES boq_files(boq_id) ON DELETE CASCADE,
    item_code VARCHAR(50),
    item_description TEXT NOT NULL,
    unit_of_measurement VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,3) NOT NULL CHECK (quantity >= 0),
    supply_unit_rate DECIMAL(18,2) CHECK (supply_unit_rate >= 0),
    supply_amount DECIMAL(18,2) GENERATED ALWAYS AS (quantity * supply_unit_rate) STORED,
    labour_unit_rate DECIMAL(18,2) CHECK (labour_unit_rate >= 0),
    labour_amount DECIMAL(18,2) GENERATED ALWAYS AS (quantity * labour_unit_rate) STORED,
    total_amount DECIMAL(18,2) GENERATED ALWAYS AS ((quantity * supply_unit_rate) + (quantity * labour_unit_rate)) STORED,
    location_id INT REFERENCES locations(location_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- ALTER TABLE boq_items 
-- ADD COLUMN description_embedding vector(768);

-- ALTER TABLE projects
-- ADD COLUMN project_name_embedding vector(768);

-- CREATE INDEX ON boq_items USING ivfflat (description_embedding vector_cosine_ops)
-- WITH (lists = 100);

-- CREATE INDEX ON projects USING ivfflat (project_name_embedding vector_cosine_ops)
-- WITH (lists = 100);

-- ALTER TABLE boq_items 
-- ADD COLUMN embedding_model VARCHAR(100),
-- ADD COLUMN embedding_generated_at TIMESTAMP WITH TIME ZONE;

-- ALTER TABLE projects 
-- ADD COLUMN embedding_model VARCHAR(100),
-- ADD COLUMN embedding_generated_at TIMESTAMP WITH TIME ZONE;
