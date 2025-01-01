-- Reset cancer table
DROP TABLE IF EXISTS cancer;

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