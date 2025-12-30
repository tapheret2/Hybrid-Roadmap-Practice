-- ================================================================
-- SE INTERN - LESSON 7: SQL FUNDAMENTALS
-- ================================================================

-- --- 1. THEORY ---
/*
1. SQL = Structured Query Language
   - DDL (Data Definition): CREATE, ALTER, DROP
   - DML (Data Manipulation): SELECT, INSERT, UPDATE, DELETE
   - DCL (Data Control): GRANT, REVOKE

2. Keys:
   - Primary Key: Unique identifier for each row
   - Foreign Key: Reference to another table's primary key
   - Unique: No duplicates allowed

3. Relationships:
   - One-to-One: User → Profile
   - One-to-Many: User → Orders
   - Many-to-Many: Students ↔ Courses (via junction table)

4. JOIN Types:
   - INNER JOIN: Only matching rows
   - LEFT JOIN: All from left + matching from right
   - RIGHT JOIN: All from right + matching from left
   - FULL JOIN: All from both tables
*/

-- --- 2. CODE SAMPLE ---

-- ========== CREATE TABLES ==========

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL CHECK (price > 0),
    stock INT DEFAULT 0 CHECK (stock >= 0),
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table (One-to-Many with users)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order items (Many-to-Many between orders and products)
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id INT NOT NULL REFERENCES products(id),
    quantity INT NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL
);

-- ========== INSERT DATA ==========

INSERT INTO users (username, email, password_hash) VALUES
    ('john_doe', 'john@example.com', 'hash123'),
    ('jane_smith', 'jane@example.com', 'hash456'),
    ('bob_wilson', 'bob@example.com', 'hash789');

INSERT INTO products (name, price, stock, category) VALUES
    ('Laptop Pro', 1299.99, 50, 'Electronics'),
    ('Wireless Mouse', 49.99, 200, 'Electronics'),
    ('Python Book', 39.99, 100, 'Books'),
    ('USB Cable', 9.99, 500, 'Accessories');

INSERT INTO orders (user_id, status, total_amount) VALUES
    (1, 'completed', 1349.98),
    (1, 'pending', 49.99),
    (2, 'completed', 89.98);

-- ========== SELECT QUERIES ==========

-- Basic SELECT
SELECT * FROM products;
SELECT name, price FROM products WHERE price > 30;

-- Sorting and limiting
SELECT name, price 
FROM products 
ORDER BY price DESC 
LIMIT 5;

-- Filtering with conditions
SELECT * FROM products 
WHERE category = 'Electronics' 
  AND price BETWEEN 10 AND 100;

-- Pattern matching
SELECT * FROM products WHERE name LIKE '%Pro%';
SELECT * FROM users WHERE email LIKE '%@example.com';

-- ========== AGGREGATE FUNCTIONS ==========

SELECT 
    COUNT(*) as total_products,
    AVG(price) as avg_price,
    SUM(stock) as total_stock,
    MAX(price) as max_price,
    MIN(price) as min_price
FROM products;

-- GROUP BY
SELECT 
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
GROUP BY category
HAVING COUNT(*) > 1
ORDER BY avg_price DESC;

-- ========== JOIN QUERIES ==========

-- INNER JOIN: Get orders with user info
SELECT 
    o.id as order_id,
    u.username,
    o.status,
    o.total_amount,
    o.created_at
FROM orders o
INNER JOIN users u ON o.user_id = u.id;

-- LEFT JOIN: All users with their order count (including users with 0 orders)
SELECT 
    u.username,
    COUNT(o.id) as order_count,
    COALESCE(SUM(o.total_amount), 0) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- Multiple JOINs
SELECT 
    u.username,
    p.name as product_name,
    oi.quantity,
    oi.unit_price
FROM order_items oi
JOIN orders o ON oi.order_id = o.id
JOIN users u ON o.user_id = u.id
JOIN products p ON oi.product_id = p.id;

-- ========== UPDATE & DELETE ==========

-- Update single row
UPDATE products 
SET price = 44.99, stock = stock - 1 
WHERE id = 2;

-- Update with condition
UPDATE orders 
SET status = 'shipped' 
WHERE status = 'pending' AND created_at < NOW() - INTERVAL '1 day';

-- Delete
DELETE FROM order_items WHERE quantity = 0;

-- ========== SUBQUERIES ==========

-- Users who have placed orders
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders);

-- Products priced above average
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- --- 3. EXERCISES ---
/*
EXERCISE 1: Write queries for:
           - Find the most expensive product
           - List products with stock below 20
           - Get total revenue per user

EXERCISE 2: Create a reviews table:
           - user_id, product_id, rating (1-5), comment
           - Query average rating per product
           - Get products with rating >= 4

EXERCISE 3: Complex queries:
           - Find users who haven't ordered in 30 days
           - Get top 3 selling products
           - Calculate month-over-month revenue

EXERCISE 4: Write UPDATE/DELETE:
           - Soft delete (set is_active = false)
           - Bulk price increase by 10%
           - Archive old orders
*/
