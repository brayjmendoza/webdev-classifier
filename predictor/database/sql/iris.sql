-- Reset iris table
DROP TABLE IF EXISTS iris;

CREATE TABLE iris (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sepallen FLOAT NOT NULL,
    sepalwid FLOAT NOT NULL,
    petallen FLOAT NOT NULL,
    petalwid FLOAT NOT NULL,
    species INT NOT NULL
);