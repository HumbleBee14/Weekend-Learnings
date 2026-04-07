-- Exercise 01: Basic Queries
-- Goal: Practice SELECT, WHERE, ORDER BY, LIMIT
--
-- Run with: sqlite3 practice.db < queries.sql

-- Setup: Create a sample table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER,
    city TEXT
);

INSERT OR IGNORE INTO users (id, name, email, age, city) VALUES
(1, 'Alice', 'alice@example.com', 30, 'Mumbai'),
(2, 'Bob', 'bob@example.com', 25, 'Delhi'),
(3, 'Charlie', 'charlie@example.com', 35, 'Bangalore'),
(4, 'Diana', 'diana@example.com', 28, 'Mumbai'),
(5, 'Eve', 'eve@example.com', 32, 'Chennai');

-- YOUR QUERIES BELOW

-- Q1: Select all users

-- Q2: Select users from Mumbai

-- Q3: Select users older than 28, ordered by age descending

-- Q4: Select the 3 youngest users
