-- =====================================================
-- COMPANY TABLE
-- =====================================================
CREATE TABLE company (
    cmp_id SERIAL PRIMARY KEY,
    cmp_name VARCHAR(145) NOT NULL,
    cmp_addr VARCHAR(45),
    fullname VARCHAR(145),
    cmp_phone VARCHAR(45),
    username VARCHAR(45),
    password VARCHAR(145),
    access_otp INT,
    cmp_status SMALLINT,
    isenable SMALLINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- PROJECTS TABLE
-- =====================================================
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name TEXT NOT NULL,
    project_code VARCHAR(50) UNIQUE NOT NULL,
    cmp_id INT REFERENCES company(cmp_id) ON DELETE CASCADE,
    client_name TEXT,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    project_name_embedding vector(768),
    embedding_model VARCHAR(100),
    embedding_generated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_projects_name_embedding
ON projects
USING ivfflat (project_name_embedding vector_cosine_ops)
WITH (lists = 100);

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
CREATE TABLE store_boq_files (
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
-- BOQ_ITEMS TABLE (FINALIZED/ESTIMATED)
-- =====================================================
CREATE TABLE store_boq_items (
    item_id SERIAL PRIMARY KEY,
    boq_id INT NOT NULL REFERENCES store_boq_files(boq_id) ON DELETE CASCADE,
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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description_embedding vector(768),
    embedding_model VARCHAR(100),
    embedding_generated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_boq_items_description_embedding
ON store_boq_items
USING ivfflat (description_embedding vector_cosine_ops)
WITH (lists = 100);

-- =====================================================
-- TO_BE_ESTIMATED_BOQ FILES TABLE
-- =====================================================
CREATE TABLE to_be_estimated_boq_files (
    boq_id SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    boq_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT,
    embedding_model VARCHAR(100),
    embedding_generated_at TIMESTAMP WITH TIME ZONE
);

-- =====================================================
-- TO_BE_ESTIMATED_BOQ ITEMS TABLE
-- =====================================================
CREATE TABLE to_be_estimated_boq_items (
    item_id SERIAL PRIMARY KEY,
    boq_id INT NOT NULL REFERENCES to_be_estimated_boq_files(boq_id) ON DELETE CASCADE,
    item_code VARCHAR(50),
    item_description TEXT NOT NULL,
    unit_of_measurement VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,3) NOT NULL CHECK (quantity >= 0),
    location_id INT REFERENCES locations(location_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description_embedding vector(768),
    embedding_model VARCHAR(100),
    embedding_generated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_to_be_estimated_boq_items_description_embedding
ON to_be_estimated_boq_items
USING ivfflat (description_embedding vector_cosine_ops)
WITH (lists = 100);
