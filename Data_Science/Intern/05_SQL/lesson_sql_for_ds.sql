-- ================================================================
-- DS INTERN - LESSON 6: SQL FOR DATA SCIENCE
-- ================================================================
-- SQL is a MUST-HAVE skill for Data Scientists

-- --- 1. THEORY ---
/*
1. SELECT: Retrieve data
2. WHERE: Filter conditions
3. GROUP BY: Aggregate by groups
4. HAVING: Filter after GROUP BY
5. JOIN: Combine multiple tables
6. Window Functions: Calculate over "windows" of data
7. Subqueries: Nested queries
*/

-- --- 2. CODE SAMPLE ---

-- Create sample tables
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(50),
    join_date DATE
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    product VARCHAR(100),
    amount DECIMAL(10, 2),
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Insert sample data
INSERT INTO customers VALUES 
    (1, 'Alice', 'New York', '2023-01-15'),
    (2, 'Bob', 'Los Angeles', '2023-02-20'),
    (3, 'Charlie', 'New York', '2023-03-10'),
    (4, 'Diana', 'Chicago', '2023-04-05');

INSERT INTO orders VALUES
    (101, 1, 'Laptop', 1500, '2024-01-10'),
    (102, 1, 'Mouse', 50, '2024-01-15'),
    (103, 2, 'Keyboard', 100, '2024-02-01'),
    (104, 2, 'Laptop', 1600, '2024-02-10'),
    (105, 3, 'Monitor', 300, '2024-02-15'),
    (106, 1, 'Headphone', 200, '2024-03-01');

-- BASIC SELECT
SELECT * FROM customers;
SELECT name, city FROM customers WHERE city = 'New York';

-- AGGREGATE FUNCTIONS
SELECT 
    COUNT(*) as total_orders,
    SUM(amount) as total_revenue,
    AVG(amount) as avg_order,
    MAX(amount) as max_order,
    MIN(amount) as min_order
FROM orders;

-- GROUP BY with HAVING
SELECT 
    customer_id,
    COUNT(*) as order_count,
    SUM(amount) as total_spent
FROM orders
GROUP BY customer_id
HAVING SUM(amount) > 500
ORDER BY total_spent DESC;

-- JOIN
SELECT 
    c.name,
    c.city,
    COUNT(o.order_id) as orders,
    COALESCE(SUM(o.amount), 0) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.city;

-- WINDOW FUNCTIONS (VERY IMPORTANT)
SELECT 
    o.order_id,
    c.name,
    o.amount,
    -- Running total
    SUM(o.amount) OVER (ORDER BY o.order_date) as running_total,
    -- Total per customer
    SUM(o.amount) OVER (PARTITION BY o.customer_id) as customer_total,
    -- Rank within customer
    RANK() OVER (PARTITION BY o.customer_id ORDER BY o.amount DESC) as rank_in_customer,
    -- Lag/Lead
    LAG(o.amount, 1) OVER (ORDER BY o.order_date) as prev_amount,
    LEAD(o.amount, 1) OVER (ORDER BY o.order_date) as next_amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- SUBQUERY
-- Customers with orders above average
SELECT name FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders
    WHERE amount > (SELECT AVG(amount) FROM orders)
);

-- CTE (Common Table Expression)
WITH customer_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT 
    c.name,
    cs.order_count,
    cs.total_spent
FROM customers c
JOIN customer_stats cs ON c.customer_id = cs.customer_id
WHERE cs.total_spent > 1000;

-- --- 3. EXERCISES ---
/*
EXERCISE 1: Find top 3 customers by total spending

EXERCISE 2: Calculate monthly revenue with month-over-month growth

EXERCISE 3: Find the most purchased product

EXERCISE 4: For each customer, find their largest order

EXERCISE 5 (Advanced): Calculate retention rate - 
           customers who purchased again in the following month
*/
