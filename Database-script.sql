CREATE TABLE plant_features (
    id          SERIAL PRIMARY KEY,
    file_name   VARCHAR(255) NOT NULL ,
    folder      VARCHAR(100) NOT NULL,

    -- RGB Histogram
    rgb_hist_1  NUMERIC(10,6), rgb_hist_2  NUMERIC(10,6),
    rgb_hist_3  NUMERIC(10,6), rgb_hist_4  NUMERIC(10,6),
    rgb_hist_5  NUMERIC(10,6), rgb_hist_6  NUMERIC(10,6),
    rgb_hist_7  NUMERIC(10,6), rgb_hist_8  NUMERIC(10,6),
    rgb_hist_9  NUMERIC(10,6), rgb_hist_10 NUMERIC(10,6),
    rgb_hist_11 NUMERIC(10,6), rgb_hist_12 NUMERIC(10,6),
    rgb_hist_13 NUMERIC(10,6), rgb_hist_14 NUMERIC(10,6),
    rgb_hist_15 NUMERIC(10,6), rgb_hist_16 NUMERIC(10,6),
    rgb_hist_17 NUMERIC(10,6), rgb_hist_18 NUMERIC(10,6),
    rgb_hist_19 NUMERIC(10,6), rgb_hist_20 NUMERIC(10,6),
    rgb_hist_21 NUMERIC(10,6), rgb_hist_22 NUMERIC(10,6),
    rgb_hist_23 NUMERIC(10,6), rgb_hist_24 NUMERIC(10,6),

    -- Edge
    edge_count  INT,
    edge_ratio  NUMERIC(10,6),

    -- Shape
    hw_ratio     NUMERIC(10,6),
    contour_area INT,
    convexity    NUMERIC(10,6),
    defect_count INT,

    -- LBP Texture
    lbp_1  NUMERIC(10,6), lbp_2  NUMERIC(10,6),
    lbp_3  NUMERIC(10,6), lbp_4  NUMERIC(10,6),
    lbp_5  NUMERIC(10,6), lbp_6  NUMERIC(10,6),
    lbp_7  NUMERIC(10,6), lbp_8  NUMERIC(10,6),
    lbp_9  NUMERIC(10,6), lbp_10 NUMERIC(10,6),

    -- Green
    green_count INT,
    green_ratio NUMERIC(10,6)
);