-- =============================================================================
-- SMART INVENTORY MANAGEMENT SYSTEM - SUPABASE SCHEMA MIGRATION
-- Modules: Feature 2 (Barcode/Expiry Verification) & Feature 3 (Visual Shelf & Damage)
-- Target Database: Supabase Cloud PostgreSQL
-- =============================================================================

-- 1. Ensure extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Shelf Inventory Table
-- Connects real warehouse shelf rows (A1, A2, B1, B2, etc.) to product stock
CREATE TABLE IF NOT EXISTS public.shelf_inventory (
    id BIGSERIAL PRIMARY KEY,
    shelf_id VARCHAR(32) NOT NULL,
    zone VARCHAR(128) DEFAULT 'Main Retail Floor',
    product_id BIGINT REFERENCES public.products(id) ON DELETE CASCADE,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(128) NOT NULL,
    barcode VARCHAR(64),
    batch_no VARCHAR(128) NOT NULL,
    recorded_quantity INTEGER NOT NULL DEFAULT 0 CHECK (recorded_quantity >= 0),
    available_quantity INTEGER NOT NULL DEFAULT 0 CHECK (available_quantity >= 0),
    damaged_quantity INTEGER NOT NULL DEFAULT 0 CHECK (damaged_quantity >= 0),
    mfg_date DATE,
    expiry_date DATE,
    expiry_status VARCHAR(64) DEFAULT 'BEST',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_shelf_product_batch UNIQUE (shelf_id, product_id, batch_no)
);

CREATE INDEX IF NOT EXISTS idx_shelf_inventory_shelf_id ON public.shelf_inventory(shelf_id);
CREATE INDEX IF NOT EXISTS idx_shelf_inventory_barcode ON public.shelf_inventory(barcode);

-- 3. Damage Events Audit Trail Table
-- Records confirmed package damage events with unique idempotency protection
CREATE TABLE IF NOT EXISTS public.damage_events (
    id BIGSERIAL PRIMARY KEY,
    damage_event_id VARCHAR(128) UNIQUE NOT NULL,
    shelf_id VARCHAR(32) NOT NULL,
    product_id BIGINT REFERENCES public.products(id) ON DELETE SET NULL,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(128) NOT NULL,
    quantity_reduced INTEGER NOT NULL DEFAULT 1 CHECK (quantity_reduced > 0),
    defect_type VARCHAR(128) DEFAULT 'Torn / Crushed Packaging',
    confidence DOUBLE PRECISION DEFAULT 1.0,
    physically_removed BOOLEAN DEFAULT TRUE,
    reason TEXT NOT NULL,
    authorized_by VARCHAR(128) DEFAULT 'Supervisor',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_damage_events_event_id ON public.damage_events(damage_event_id);
CREATE INDEX IF NOT EXISTS idx_damage_events_shelf ON public.damage_events(shelf_id);

-- 4. Stock Movement Audit Trail Table
-- Tracks transfers between shelves and adjustments
CREATE TABLE IF NOT EXISTS public.stock_movements (
    id BIGSERIAL PRIMARY KEY,
    movement_id VARCHAR(128) UNIQUE NOT NULL DEFAULT uuid_generate_v4()::text,
    product_id BIGINT REFERENCES public.products(id) ON DELETE SET NULL,
    product_name VARCHAR(255),
    from_shelf VARCHAR(32),
    to_shelf VARCHAR(32),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    movement_type VARCHAR(64) NOT NULL, -- 'TRANSFER', 'BARCODE_ADDITION', 'DAMAGE_REMOVAL', 'QUARANTINE'
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Atomic Damaged Stock Deduction Function
-- Safeguards: Prevents double deductions (idempotent), prevents negative inventory
CREATE OR REPLACE FUNCTION public.deduct_damaged_stock(
    p_damage_event_id VARCHAR,
    p_shelf_id VARCHAR,
    p_product_id BIGINT,
    p_quantity INTEGER DEFAULT 1,
    p_reason TEXT DEFAULT 'Physically damaged package removed from shelf',
    p_authorized_by VARCHAR DEFAULT 'Supervisor',
    p_physically_removed BOOLEAN DEFAULT TRUE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_existing_event RECORD;
    v_shelf_stock RECORD;
    v_new_recorded INTEGER;
    v_new_available INTEGER;
BEGIN
    -- Idempotency check: prevent duplicate damage event deduction
    SELECT * INTO v_existing_event FROM public.damage_events WHERE damage_event_id = p_damage_event_id;
    IF FOUND THEN
        RETURN jsonb_build_object(
            'success', FALSE,
            'error', 'Damage event already processed. Duplicate deduction prevented.',
            'damage_event_id', p_damage_event_id
        );
    END IF;

    -- Fetch current shelf stock
    SELECT * INTO v_shelf_stock FROM public.shelf_inventory 
    WHERE shelf_id = p_shelf_id AND product_id = p_product_id 
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'success', FALSE,
            'error', 'Product not found on specified shelf.'
        );
    END IF;

    -- Calculate non-negative new quantities
    IF p_physically_removed THEN
        v_new_recorded := GREATEST(0, v_shelf_stock.recorded_quantity - p_quantity);
        v_new_available := GREATEST(0, v_shelf_stock.available_quantity - p_quantity);
        
        UPDATE public.shelf_inventory
        SET recorded_quantity = v_new_recorded,
            available_quantity = v_new_available,
            damaged_quantity = v_shelf_stock.damaged_quantity + p_quantity,
            updated_at = NOW()
        WHERE id = v_shelf_stock.id;
    ELSE
        v_new_available := GREATEST(0, v_shelf_stock.available_quantity - p_quantity);
        
        UPDATE public.shelf_inventory
        SET available_quantity = v_new_available,
            damaged_quantity = v_shelf_stock.damaged_quantity + p_quantity,
            updated_at = NOW()
        WHERE id = v_shelf_stock.id;
    END IF;

    -- Insert into damage audit log
    INSERT INTO public.damage_events (
        damage_event_id, shelf_id, product_id, product_name, category,
        quantity_reduced, physically_removed, reason, authorized_by
    ) VALUES (
        p_damage_event_id, p_shelf_id, p_product_id, v_shelf_stock.product_name, v_shelf_stock.category,
        p_quantity, p_physically_removed, p_reason, p_authorized_by
    );

    RETURN jsonb_build_object(
        'success', TRUE,
        'message', 'Damaged package stock successfully deducted and logged.',
        'damage_event_id', p_damage_event_id,
        'new_recorded_qty', v_new_recorded,
        'new_available_qty', v_new_available
    );
END;
$$;

-- 6. Row Level Security (RLS) Configuration
-- Grants necessary read & write permissions to anon/authenticated roles
ALTER TABLE public.shelf_inventory ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.damage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.stock_movements ENABLE ROW LEVEL SECURITY;

-- Allow public read access to shelves
DROP POLICY IF EXISTS "Public read shelf_inventory" ON public.shelf_inventory;
CREATE POLICY "Public read shelf_inventory" ON public.shelf_inventory FOR SELECT USING (true);

DROP POLICY IF EXISTS "Public insert/update shelf_inventory" ON public.shelf_inventory;
CREATE POLICY "Public insert/update shelf_inventory" ON public.shelf_inventory FOR ALL USING (true);

-- Allow public read/insert access to damage events
DROP POLICY IF EXISTS "Public access damage_events" ON public.damage_events;
CREATE POLICY "Public access damage_events" ON public.damage_events FOR ALL USING (true);

-- Allow public read/insert access to movements
DROP POLICY IF EXISTS "Public access stock_movements" ON public.stock_movements;
CREATE POLICY "Public access stock_movements" ON public.stock_movements FOR ALL USING (true);

-- Ensure public anon role can access RPC functions
GRANT EXECUTE ON FUNCTION public.deduct_damaged_stock TO anon, authenticated;
GRANT ALL ON TABLE public.shelf_inventory TO anon, authenticated;
GRANT ALL ON TABLE public.damage_events TO anon, authenticated;
GRANT ALL ON TABLE public.stock_movements TO anon, authenticated;
