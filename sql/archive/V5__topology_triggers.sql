-- ============================================================
-- A. Block QUESTION → QUESTION edges
-- ============================================================
CREATE OR REPLACE FUNCTION block_qq_links() RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT label FROM nodes WHERE id = NEW.source_id) = 'QUESTION'
       AND (SELECT label FROM nodes WHERE id = NEW.target_id) = 'QUESTION' THEN
        RAISE EXCEPTION 'Questions cannot connect to other questions';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_01_block_qq
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION block_qq_links();

-- ============================================================
-- B. 5-Slot answer eviction with edge migration
-- ============================================================
CREATE OR REPLACE FUNCTION enforce_answer_slots() RETURNS TRIGGER AS $$
DECLARE
    min_conf FLOAT;
    old_answer_id UUID;
    new_conf FLOAT;
BEGIN
    IF NEW.label != 'ANSWERS' THEN RETURN NEW; END IF;

    -- Slot available
    IF (SELECT COUNT(*) FROM edges WHERE target_id = NEW.target_id AND label = 'ANSWERS') < 5 THEN
        UPDATE nodes SET degree = degree + 1 WHERE id = NEW.target_id;
        RETURN NEW;
    END IF;

    -- Saturated: find lowest-confidence existing answer
    SELECT a.id, a.confidence INTO old_answer_id, min_conf
    FROM edges e JOIN nodes a ON e.source_id = a.id
    WHERE e.target_id = NEW.target_id AND e.label = 'ANSWERS'
    ORDER BY a.confidence ASC, a.created_at ASC LIMIT 1;

    SELECT confidence INTO new_conf FROM nodes WHERE id = NEW.source_id;

    IF new_conf <= min_conf THEN
        RAISE EXCEPTION 'Answer rejected: confidence % <= %', new_conf, min_conf;
    END IF;

    -- Evict: expire old answer
    UPDATE nodes SET valid_until = NOW() WHERE id = old_answer_id;

    -- Record supersession
    INSERT INTO edges (source_id, target_id, label)
    VALUES (NEW.source_id, old_answer_id, 'SUPERSEDES');

    -- Migrate weaving edges from evicted answer to new answer
    UPDATE edges SET source_id = NEW.source_id
    WHERE source_id = old_answer_id AND label IN ('CONTEXT_OF', 'ELABORATES');

    UPDATE edges SET target_id = NEW.source_id
    WHERE target_id = old_answer_id AND label IN ('CONTEXT_OF', 'ELABORATES');

    -- Remove old ANSWERS edge
    DELETE FROM edges
    WHERE source_id = old_answer_id AND target_id = NEW.target_id AND label = 'ANSWERS';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_02_answer_slots
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION enforce_answer_slots();

-- ============================================================
-- C. General degree cap (PROJECT nodes exempted via degree_cap = 0)
-- ============================================================
CREATE OR REPLACE FUNCTION enforce_degree_cap() RETURNS TRIGGER AS $$
DECLARE
    src_cap INTEGER;
    src_deg INTEGER;
    tgt_cap INTEGER;
    tgt_deg INTEGER;
BEGIN
    SELECT degree_cap, degree INTO src_cap, src_deg FROM nodes WHERE id = NEW.source_id;
    SELECT degree_cap, degree INTO tgt_cap, tgt_deg FROM nodes WHERE id = NEW.target_id;

    -- Check source cap (0 = unlimited)
    IF src_cap > 0 AND src_deg >= src_cap THEN
        RAISE EXCEPTION 'Source % at degree cap %', NEW.source_id, src_cap;
    END IF;

    -- Check target cap (skip for ANSWERS — handled by enforce_answer_slots)
    IF NEW.label != 'ANSWERS' AND tgt_cap > 0 AND tgt_deg >= tgt_cap THEN
        RAISE EXCEPTION 'Target % at degree cap %', NEW.target_id, tgt_cap;
    END IF;

    -- Increment degrees
    UPDATE nodes SET degree = degree + 1 WHERE id = NEW.source_id;
    IF NEW.label != 'ANSWERS' THEN
        UPDATE nodes SET degree = degree + 1 WHERE id = NEW.target_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_03_degree_cap
BEFORE INSERT ON edges
FOR EACH ROW
EXECUTE FUNCTION enforce_degree_cap();
