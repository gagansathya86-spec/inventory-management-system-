-- =============================================================================
-- SMART INVENTORY MANAGEMENT SYSTEM - FEATURE 4 ISOLATED INVENTORY MIGRATION
-- Module: Feature 4 (Smart Trend Analysis, Top 5 Trending, Automatic Restock)
-- Target Database: Existing Supabase Cloud PostgreSQL (nnqrrvvmukgpjywjboaq)
-- Tables Created:
--   1. public.trend_inventory_products
--   2. public.trend_inventory_events
--   3. public.trend_inventory_forecasts
-- Approved Categories: Exactly 17 categories from YOLOv8 model / categories.json
-- =============================================================================

-- 1. Ensure uuid extension is available
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Table A: trend_inventory_products
-- Dedicated approved inventory dataset for Feature 4 (no legacy products)
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.trend_inventory_products (
    id BIGSERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(128) NOT NULL CHECK (category IN (
        'biscuit', 'chips', 'pen', 'scale', 'phone', 'calculator',
        'headphone', 'laptop', 'books', 'pencils', 'erasers', 'keyboard',
        'mouse', 'toothpaste', 'ball', 'toothbrush', 'wires'
    )),
    barcode VARCHAR(64) UNIQUE,
    current_stock INTEGER CHECK (current_stock IS NULL OR current_stock >= 0),
    reorder_point INTEGER NOT NULL DEFAULT 10 CHECK (reorder_point >= 0),
    supplier_lead_time_days INTEGER NOT NULL DEFAULT 3 CHECK (supplier_lead_time_days >= 1),
    safety_stock INTEGER NOT NULL DEFAULT 5 CHECK (safety_stock >= 0),
    is_initialized BOOLEAN NOT NULL DEFAULT FALSE,
    shelf_location VARCHAR(32) DEFAULT 'S1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_trend_inv_category UNIQUE (category)
);

CREATE INDEX IF NOT EXISTS idx_trend_inv_prod_cat ON public.trend_inventory_products(category);
CREATE INDEX IF NOT EXISTS idx_trend_inv_prod_bcode ON public.trend_inventory_products(barcode);

-- =============================================================================
-- Table B: trend_inventory_events
-- Dedicated multi-source historical activity tracking for trend scoring & restock
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.trend_inventory_events (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES public.trend_inventory_products(id) ON DELETE CASCADE,
    category VARCHAR(128) NOT NULL CHECK (category IN (
        'biscuit', 'chips', 'pen', 'scale', 'phone', 'calculator',
        'headphone', 'laptop', 'books', 'pencils', 'erasers', 'keyboard',
        'mouse', 'toothpaste', 'ball', 'toothbrush', 'wires'
    )),
    event_type VARCHAR(64) NOT NULL CHECK (event_type IN (
        'stock_out', 'stock_movement', 'barcode_scan', 'restock', 'camera_detection'
    )),
    quantity_change INTEGER NOT NULL DEFAULT 0,
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(64) NOT NULL CHECK (source IN (
        'barcode_scanner', 'camera_recognition', 'shelf_inspection', 'restock_action', 'manual_audit'
    )),
    deduplication_key VARCHAR(255) UNIQUE NOT NULL,
    confirmation_status VARCHAR(64) NOT NULL DEFAULT 'confirmed',
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trend_inv_events_prod_id ON public.trend_inventory_events(product_id);
CREATE INDEX IF NOT EXISTS idx_trend_inv_events_type_time ON public.trend_inventory_events(event_type, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_trend_inv_events_dedup ON public.trend_inventory_events(deduplication_key);

-- =============================================================================
-- Table C: trend_inventory_forecasts
-- Optional persisted forecast horizon snapshots for auditability
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.trend_inventory_forecasts (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT NOT NULL REFERENCES public.trend_inventory_products(id) ON DELETE CASCADE,
    forecast_horizon_days INTEGER NOT NULL DEFAULT 7,
    predicted_demand DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(64) NOT NULL DEFAULT 'lstm_demand_forecaster_v1',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trend_inv_forecasts_prod ON public.trend_inventory_forecasts(product_id);

-- =============================================================================
-- Row Level Security (RLS) Configuration
-- =============================================================================
ALTER TABLE public.trend_inventory_products ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trend_inventory_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trend_inventory_forecasts ENABLE ROW LEVEL SECURITY;

-- Permissive policies for existing backend & authenticated clients
DROP POLICY IF EXISTS "trend_inv_prod_read" ON public.trend_inventory_products;
CREATE POLICY "trend_inv_prod_read" ON public.trend_inventory_products FOR SELECT USING (true);

DROP POLICY IF EXISTS "trend_inv_prod_write" ON public.trend_inventory_products;
CREATE POLICY "trend_inv_prod_write" ON public.trend_inventory_products FOR ALL USING (true);

DROP POLICY IF EXISTS "trend_inv_events_read" ON public.trend_inventory_events;
CREATE POLICY "trend_inv_events_read" ON public.trend_inventory_events FOR SELECT USING (true);

DROP POLICY IF EXISTS "trend_inv_events_write" ON public.trend_inventory_events;
CREATE POLICY "trend_inv_events_write" ON public.trend_inventory_events FOR ALL USING (true);

DROP POLICY IF EXISTS "trend_inv_forecasts_read" ON public.trend_inventory_forecasts;
CREATE POLICY "trend_inv_forecasts_read" ON public.trend_inventory_forecasts FOR SELECT USING (true);

DROP POLICY IF EXISTS "trend_inv_forecasts_write" ON public.trend_inventory_forecasts;
CREATE POLICY "trend_inv_forecasts_write" ON public.trend_inventory_forecasts FOR ALL USING (true);

-- =============================================================================
-- Idempotent Seed: Exact 17 Approved Categories Only
-- Stock is initialized as uninitialized (NULL / false) until genuine input
-- =============================================================================
INSERT INTO public.trend_inventory_products (
    product_name, category, current_stock, reorder_point, supplier_lead_time_days, safety_stock, is_initialized, shelf_location
) VALUES
('Biscuit', 'biscuit', NULL, 10, 2, 4, false, 'A1'),
('Chips', 'chips', NULL, 12, 2, 5, false, 'A2'),
('Pen', 'pen', NULL, 15, 4, 6, false, 'B1'),
('Scale', 'scale', NULL, 8, 4, 3, false, 'B1'),
('Phone', 'phone', NULL, 5, 7, 2, false, 'C2'),
('Calculator', 'calculator', NULL, 6, 6, 3, false, 'C2'),
('Headphone', 'headphone', NULL, 8, 5, 3, false, 'C2'),
('Laptop', 'laptop', NULL, 4, 7, 2, false, 'C2'),
('Books', 'books', NULL, 14, 4, 5, false, 'B2'),
('Pencils', 'pencils', NULL, 15, 3, 6, false, 'B2'),
('Erasers', 'erasers', NULL, 12, 3, 4, false, 'B3'),
('Keyboard', 'keyboard', NULL, 6, 5, 3, false, 'C2'),
('Mouse', 'mouse', NULL, 8, 5, 3, false, 'C2'),
('Toothpaste', 'toothpaste', NULL, 10, 3, 4, false, 'C1'),
('Ball', 'ball', NULL, 8, 4, 3, false, 'A3'),
('Toothbrush', 'toothbrush', NULL, 10, 3, 4, false, 'C1'),
('Wires', 'wires', NULL, 10, 5, 4, false, 'C3')
ON CONFLICT (category) DO UPDATE SET
    reorder_point = EXCLUDED.reorder_point,
    supplier_lead_time_days = EXCLUDED.supplier_lead_time_days,
    safety_stock = EXCLUDED.safety_stock,
    updated_at = NOW();

-- =============================================================================
-- REVERSIBLE DOWN MIGRATION (ROLLBACK INSTRUCTIONS)
-- Run the following commands if rollback is ever required:
--
-- DROP TABLE IF EXISTS public.trend_inventory_forecasts CASCADE;
-- DROP TABLE IF EXISTS public.trend_inventory_events CASCADE;
-- DROP TABLE IF EXISTS public.trend_inventory_products CASCADE;
-- =============================================================================
