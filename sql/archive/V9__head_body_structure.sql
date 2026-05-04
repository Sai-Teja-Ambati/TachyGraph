-- Enforce summary (head) exists on SUMMARY and ANSWER nodes
-- CHUNKs are raw text blocks — they get linked to SUMMARY nodes via edges, not via their own summary field
ALTER TABLE nodes ADD CONSTRAINT chk_summary_required
    CHECK (
        label NOT IN ('SUMMARY', 'ANSWER')
        OR summary IS NOT NULL
    );
