-- Auto-set 5-day validity window on insert if valid_until is null
CREATE OR REPLACE FUNCTION set_temporal_window()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.valid_from IS NOT NULL AND NEW.valid_until IS NULL THEN
        NEW.valid_until := NEW.valid_from + INTERVAL '5 days';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_temporal_window
BEFORE INSERT ON nodes
FOR EACH ROW EXECUTE FUNCTION set_temporal_window();

-- Extend validity when a fact is reaffirmed
CREATE OR REPLACE FUNCTION extend_validity(target_node_id UUID, extension_days INT DEFAULT 5)
RETURNS VOID AS $$
BEGIN
    UPDATE nodes
    SET valid_until = GREATEST(valid_until, NOW() + (extension_days || ' days')::INTERVAL)
    WHERE id = target_node_id;
END;
$$ LANGUAGE plpgsql;

-- Check if node is within 24h of expiration
CREATE OR REPLACE FUNCTION is_expiring_soon(target_node_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    node_valid_until TIMESTAMP;
BEGIN
    SELECT valid_until INTO node_valid_until FROM nodes WHERE id = target_node_id;
    RETURN node_valid_until IS NOT NULL
       AND node_valid_until > NOW()
       AND node_valid_until <= NOW() + INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;
