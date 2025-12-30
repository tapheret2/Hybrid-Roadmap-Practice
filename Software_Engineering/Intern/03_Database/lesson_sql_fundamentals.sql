-- ================================================================
-- SE INTERN - DATABASE: SQL FUNDAMENTALS
-- ================================================================
-- Sử dụng: PostgreSQL hoặc MySQL
-- Tools: pgAdmin, DBeaver, hoặc bất kỳ SQL client nào

-- --- 1. LÝ THUYẾT (THEORY) ---
/*
1. DDL (Data Definition Language): CREATE, ALTER, DROP
2. DML (Data Manipulation Language): SELECT, INSERT, UPDATE, DELETE
3. Primary Key: Định danh duy nhất cho mỗi row
4. Foreign Key: Liên kết giữa các bảng
5. JOIN: Kết hợp dữ liệu từ nhiều bảng
6. Index: Tăng tốc truy vấn (đánh đổi bằng tốc độ write)
*/

-- --- 2. CODE MẪU (CODE SAMPLE) ---

-- Tạo bảng
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- Auto-increment (PostgreSQL)
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(50),
    in_stock BOOLEAN DEFAULT true
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),  -- Foreign Key
    product_id INT REFERENCES products(id),
    quantity INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- INSERT
INSERT INTO users (name, email) VALUES 
    ('An', 'an@example.com'),
    ('Bình', 'binh@example.com'),
    ('Chi', 'chi@example.com');

INSERT INTO products (name, price, category) VALUES
    ('iPhone 15', 999.99, 'electronics'),
    ('MacBook Pro', 2499.99, 'electronics'),
    ('T-Shirt', 29.99, 'clothing');

INSERT INTO orders (user_id, product_id, quantity) VALUES
    (1, 1, 2),
    (1, 3, 1),
    (2, 2, 1);

-- SELECT cơ bản
SELECT * FROM users;
SELECT name, email FROM users WHERE name LIKE 'A%';
SELECT * FROM products WHERE price > 100 ORDER BY price DESC;

-- Aggregate functions
SELECT COUNT(*) FROM users;
SELECT category, AVG(price) as avg_price FROM products GROUP BY category;
SELECT category, COUNT(*) as total FROM products GROUP BY category HAVING COUNT(*) > 1;

-- JOIN
SELECT 
    u.name as customer,
    p.name as product,
    o.quantity,
    (p.price * o.quantity) as total
FROM orders o
INNER JOIN users u ON o.user_id = u.id
INNER JOIN products p ON o.product_id = p.id;

-- LEFT JOIN (bao gồm users chưa có order)
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- UPDATE
UPDATE products SET price = price * 0.9 WHERE category = 'clothing';

-- DELETE
DELETE FROM orders WHERE quantity < 1;

-- --- 3. BÀI TẬP (EXERCISE) ---
/*
BÀI 1: Viết query lấy tất cả users có ít nhất 1 order

BÀI 2: Viết query tính tổng doanh thu (price * quantity) theo từng category

BÀI 3: Viết query tìm user có tổng chi tiêu cao nhất

BÀI 4: Tạo bảng "reviews" với fields:
       - id, user_id (FK), product_id (FK), rating (1-5), comment, created_at
       Viết query lấy trung bình rating của mỗi product

BÀI 5 (Nâng cao): Viết query lấy top 3 products bán chạy nhất (theo số lượng)
*/
