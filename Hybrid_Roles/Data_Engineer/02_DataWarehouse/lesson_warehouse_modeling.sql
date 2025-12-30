-- ================================================================
-- DATA ENGINEER - MODULE 2: DATA WAREHOUSE & MODELING
-- ================================================================
-- Data Warehouse là nơi lưu trữ dữ liệu đã được transform để analytics

-- --- 1. LÝ THUYẾT (THEORY) ---
/*
1. Data Warehouse vs Database:
   - Database (OLTP): Optimize cho transactions (INSERT, UPDATE)
   - Warehouse (OLAP): Optimize cho analytics (SELECT, aggregations)

2. Data Modeling Approaches:
   - Star Schema: Fact tables + Dimension tables
   - Snowflake Schema: Normalized dimensions
   - Data Vault: Hub, Link, Satellite (for enterprise)

3. Key Concepts:
   - Fact Table: Chứa metrics (sales_amount, quantity)
   - Dimension Table: Chứa attributes (product_name, customer_info)
   - Slowly Changing Dimensions (SCD): Track history

4. Modern Data Stack:
   - Snowflake, BigQuery, Redshift: Cloud warehouses
   - dbt: Transform layer
   - Fivetran/Airbyte: ELT ingestion
*/

-- --- 2. CODE MẪU (CODE SAMPLE) ---

-- ========== STAR SCHEMA DESIGN ==========

-- Dimension: Date
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_week VARCHAR(10),
    day_of_month INT,
    month INT,
    month_name VARCHAR(10),
    quarter INT,
    year INT,
    is_weekend BOOLEAN
);

-- Dimension: Product
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(200),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    unit_cost DECIMAL(10, 2),
    -- SCD Type 2 fields
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN
);

-- Dimension: Customer
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    customer_name VARCHAR(200),
    email VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(100),
    segment VARCHAR(50),  -- B2B, B2C, Enterprise
    -- SCD Type 2
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN
);

-- Fact: Sales
CREATE TABLE fact_sales (
    sales_key BIGINT PRIMARY KEY,
    date_key INT REFERENCES dim_date(date_key),
    product_key INT REFERENCES dim_product(product_key),
    customer_key INT REFERENCES dim_customer(customer_key),
    -- Measures
    quantity INT,
    unit_price DECIMAL(10, 2),
    discount_amount DECIMAL(10, 2),
    sales_amount DECIMAL(12, 2),
    profit DECIMAL(12, 2),
    -- Degenerate dimension
    order_id VARCHAR(50)
);

-- ========== ANALYTICAL QUERIES ==========

-- Sales by Category and Month
SELECT 
    d.year,
    d.month_name,
    p.category,
    SUM(f.sales_amount) as total_sales,
    SUM(f.quantity) as total_quantity,
    SUM(f.profit) as total_profit,
    AVG(f.discount_amount) as avg_discount
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE d.year = 2024
GROUP BY d.year, d.month, d.month_name, p.category
ORDER BY d.month, total_sales DESC;

-- Customer Segmentation Analysis
SELECT 
    c.segment,
    c.country,
    COUNT(DISTINCT c.customer_key) as customer_count,
    SUM(f.sales_amount) as total_revenue,
    SUM(f.sales_amount) / COUNT(DISTINCT c.customer_key) as revenue_per_customer
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
GROUP BY c.segment, c.country
ORDER BY total_revenue DESC;

-- Year-over-Year Growth
WITH yearly_sales AS (
    SELECT 
        d.year,
        SUM(f.sales_amount) as total_sales
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.year
)
SELECT 
    year,
    total_sales,
    LAG(total_sales) OVER (ORDER BY year) as prev_year_sales,
    (total_sales - LAG(total_sales) OVER (ORDER BY year)) / 
        LAG(total_sales) OVER (ORDER BY year) * 100 as yoy_growth
FROM yearly_sales;

-- ========== SCD TYPE 2 IMPLEMENTATION ==========

-- Cập nhật dimension khi có thay đổi
-- Step 1: Close existing record
UPDATE dim_customer
SET expiration_date = CURRENT_DATE - INTERVAL '1 day',
    is_current = FALSE
WHERE customer_id = 'C001' AND is_current = TRUE;

-- Step 2: Insert new record
INSERT INTO dim_customer (
    customer_key, customer_id, customer_name, email, city, country, segment,
    effective_date, expiration_date, is_current
) VALUES (
    (SELECT MAX(customer_key) + 1 FROM dim_customer),
    'C001', 'Alice Updated', 'alice.new@email.com', 'New City', 'Vietnam', 'Enterprise',
    CURRENT_DATE, '9999-12-31', TRUE
);

-- ========== MATERIALIZED VIEWS ==========

-- Pre-aggregate cho queries thường dùng
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT 
    d.full_date,
    p.category,
    SUM(f.sales_amount) as daily_sales,
    SUM(f.quantity) as daily_quantity,
    COUNT(*) as transaction_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY d.full_date, p.category;

-- Refresh materialized view
-- REFRESH MATERIALIZED VIEW mv_daily_sales;

-- --- 3. BÀI TẬP (EXERCISE) ---
/*
BÀI 1: Thiết kế Star Schema cho e-commerce:
       - Fact: Orders (order_id, customer_key, product_key, seller_key, date_key, ...)
       - Dimensions: Customer, Product, Seller, Date, Location

BÀI 2: Implement SCD Type 2 cho dim_product:
       - Trigger tự động close old record và insert new

BÀI 3: Viết queries:
       - Top 10 products by revenue (quarter)
       - Customer lifetime value
       - Cohort analysis (retention by signup month)

BÀI 4: Optimize warehouse:
       - Partition fact table by date
       - Create appropriate indexes
       - Implement column compression
*/
