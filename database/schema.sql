-- ========================================
-- AI Log Anomaly Detection
-- Database Schema
-- ========================================

CREATE TABLE IF NOT EXISTS log_predictions (
    id              SERIAL PRIMARY KEY,
    log_date        VARCHAR(20),
    log_time        VARCHAR(20),
    level           VARCHAR(20),
    component       VARCHAR(100),
    message         TEXT,
    event_template  TEXT,
    anomaly_status  VARCHAR(20),
    anomaly_score   DOUBLE PRECISION,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster anomaly queries
CREATE INDEX IF NOT EXISTS idx_anomaly_status
    ON log_predictions (anomaly_status);

-- Index for ordering by most recent
CREATE INDEX IF NOT EXISTS idx_id_desc
    ON log_predictions (id DESC);
