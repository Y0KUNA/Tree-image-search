-- Migration: add Hu moments columns to plant_features
-- Adds hu_1 .. hu_7 as double precision columns

BEGIN;

ALTER TABLE IF EXISTS plant_features
  ADD COLUMN IF NOT EXISTS hu_1 double precision,
  ADD COLUMN IF NOT EXISTS hu_2 double precision,
  ADD COLUMN IF NOT EXISTS hu_3 double precision,
  ADD COLUMN IF NOT EXISTS hu_4 double precision,
  ADD COLUMN IF NOT EXISTS hu_5 double precision,
  ADD COLUMN IF NOT EXISTS hu_6 double precision,
  ADD COLUMN IF NOT EXISTS hu_7 double precision;

COMMIT;
