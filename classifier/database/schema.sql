DROP TABLE IF EXISTS iris;
DROP TABLE IF EXISTS cancer;
DROP VIEW IF EXISTS cancer_features;

CREATE TABLE iris (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sepallen FLOAT NOT NULL,
    sepalwid FLOAT NOT NULL,
    petallen FLOAT NOT NULL,
    petalwid FLOAT NOT NULL,
    species INT NOT NULL,
    model VARCHAR(63) NOT NULL
);

CREATE TABLE cancer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    radius FLOAT NOT NULL,
    texture FLOAT NOT NULL,
    perimeter FLOAT NOT NULL,
    area FLOAT NOT NULL,
    smoothness FLOAT NOT NULL,
    compactness FLOAT NOT NULL,
    concavity FLOAT NOT NULL,
    concave_points FLOAT NOT NULL,
    symmetry FLOAT NOT NULL,
    fractal FLOAT NOT NULL,
    radius_err FLOAT,
    texture_err FLOAT,
    perimeter_err FLOAT,
    area_err FLOAT,
    smoothness_err FLOAT,
    compactness_err FLOAT,
    concavity_err FLOAT,
    concave_points_err FLOAT,
    symmetry_err FLOAT,
    fractal_err FLOAT,
    radius_wrst FLOAT,
    texture_wrst FLOAT,
    perimeter_wrst FLOAT,
    area_wrst FLOAT,
    smoothness_wrst FLOAT,
    compactness_wrst FLOAT,
    concavity_wrst FLOAT,
    concave_points_wrst FLOAT,
    symmetry_wrst FLOAT,
    fractal_wrst FLOAT,
    classification INT NOT NULL,
    model VARCHAR(63) NOT NULL
);

CREATE VIEW cancer_features
AS
    SELECT radius, texture, perimeter, area,
           smoothness, compactness, concavity,
           concave_points, symmetry, fractal,
           radius_err, texture_err, perimeter_err,
           area_err, smoothness_err, compactness_err,
           concavity_err, concave_points_err, 
           symmetry_err, fractal_err, radius_wrst,
           texture_wrst, perimeter_wrst, area_wrst,
           smoothness_wrst, compactness_wrst,
           concavity_wrst, concave_points_wrst,
           symmetry_wrst, fractal_wrst, model
    FROM cancer