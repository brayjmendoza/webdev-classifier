DROP TABLE IF EXISTS iris;
DROP TABLE IF EXISTS cancer;
DROP VIEW IF EXISTS cancer_data;

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
    cell_type INT NOT NULL,
    model VARCHAR(63) NOT NULL
);

CREATE VIEW cancer_data
AS
    SELECT radius, texture, perimeter, area,
           smoothness, compactness, concavity,
           concave_points, symmetry, fractal,
           cell_type, model
    FROM cancer