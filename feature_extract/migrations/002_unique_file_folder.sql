-- Migration: ensure unique constraint on (file_name, folder)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        WHERE tc.constraint_type = 'UNIQUE'
          AND tc.table_name = 'plant_features'
          AND tc.constraint_name = 'plant_features_file_folder_unique'
    ) THEN
        ALTER TABLE plant_features
        ADD CONSTRAINT plant_features_file_folder_unique UNIQUE (file_name, folder);
    END IF;
END
$$;
