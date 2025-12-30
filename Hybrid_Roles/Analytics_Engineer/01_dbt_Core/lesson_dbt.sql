-- ================================================================
-- ANALYTICS ENGINEER - MODULE 1: DBT (DATA BUILD TOOL) CORE
-- ================================================================
-- dbt là công cụ transform data trong warehouse
-- "Analytics Engineers" xây dựng data models với SQL

-- --- 1. LÝ THUYẾT (THEORY) ---
/*
1. dbt là gì?
   - Transform layer trong ELT
   - SQL-first, version controlled
   - Tự động build dependencies
   - Testing và documentation built-in

2. dbt Core Concepts:
   - Models: SQL SELECT statements → tables/views
   - Sources: Raw data tables
   - Seeds: CSV files loaded to warehouse
   - Tests: Data quality checks
   - Macros: Reusable SQL snippets (Jinja)

3. Materialization Types:
   - view: Virtual table (default)
   - table: Physical table
   - incremental: Only new/changed rows
   - ephemeral: CTE, not materialized

4. Ref và Source:
   - {{ ref('model_name') }}: Link to other model
   - {{ source('source_name', 'table') }}: Link to raw data
*/

-- --- 2. CODE MẪU (CODE SAMPLE) ---

-- ========== PROJECT STRUCTURE ==========
/*
my_dbt_project/
├── dbt_project.yml           # Project config
├── models/
│   ├── staging/              # Clean raw data
│   │   ├── stg_orders.sql
│   │   ├── stg_customers.sql
│   │   └── staging.yml       # Tests & docs
│   ├── intermediate/         # Business logic
│   │   └── int_orders_enriched.sql
│   └── marts/                # Final models for BI
│       ├── dim_customers.sql
│       ├── fct_orders.sql
│       └── marts.yml
├── seeds/                    # Static CSV data
├── macros/                   # Reusable SQL
├── tests/                    # Custom tests
└── snapshots/                # SCD Type 2
*/

-- ========== STAGING MODEL ==========
-- File: models/staging/stg_orders.sql

-- Config at top of file
-- {{ config(materialized='view') }}

-- WITH source AS (
--     SELECT * FROM {{ source('raw', 'orders') }}
-- ),

-- renamed AS (
--     SELECT
--         id AS order_id,
--         user_id AS customer_id,
--         created_at AS order_date,
--         status AS order_status,
--         total_amount_cents / 100.0 AS order_amount,
--         -- Metadata
--         _loaded_at AS loaded_at
--     FROM source
--     WHERE status != 'cancelled'
-- )

-- SELECT * FROM renamed

-- Simplified version without Jinja:
CREATE VIEW stg_orders AS
SELECT
    id AS order_id,
    user_id AS customer_id,
    created_at AS order_date,
    status AS order_status,
    total_amount_cents / 100.0 AS order_amount
FROM raw.orders
WHERE status != 'cancelled';

-- ========== INTERMEDIATE MODEL ==========
-- File: models/intermediate/int_orders_enriched.sql

CREATE VIEW int_orders_enriched AS
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    o.order_amount,
    c.customer_name,
    c.customer_segment,
    c.country,
    -- Derived metrics
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS order_sequence,
    CASE 
        WHEN ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) = 1 
        THEN TRUE ELSE FALSE 
    END AS is_first_order
FROM stg_orders o  -- {{ ref('stg_orders') }}
LEFT JOIN stg_customers c ON o.customer_id = c.customer_id;  -- {{ ref('stg_customers') }}

-- ========== MART MODEL (FACTS) ==========
-- File: models/marts/fct_orders.sql

-- {{ config(materialized='incremental', unique_key='order_id') }}

CREATE TABLE fct_orders AS
SELECT
    order_id,
    customer_id,
    order_date,
    DATE_TRUNC('month', order_date) AS order_month,
    order_amount,
    is_first_order,
    -- Aggregates
    SUM(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS customer_lifetime_value
FROM int_orders_enriched;  -- {{ ref('int_orders_enriched') }}

-- For incremental:
-- {% if is_incremental() %}
--     WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
-- {% endif %}

-- ========== TESTING (schema.yml) ==========
/*
version: 2

models:
  - name: fct_orders
    description: "Fact table containing all orders"
    columns:
      - name: order_id
        description: "Primary key"
        tests:
          - unique
          - not_null
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: order_amount
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
*/

-- ========== MACROS ==========
-- File: macros/cents_to_dollars.sql
/*
{% macro cents_to_dollars(column_name) %}
    ({{ column_name }} / 100.0)
{% endmacro %}

-- Usage in model:
SELECT
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM orders
*/

-- ========== DBT COMMANDS ==========
/*
# Install
pip install dbt-postgres  # or dbt-snowflake, dbt-bigquery

# Initialize project
dbt init my_project

# Run all models
dbt run

# Run specific model
dbt run --select fct_orders

# Run with upstream dependencies
dbt run --select +fct_orders

# Test
dbt test

# Generate docs
dbt docs generate
dbt docs serve

# Full refresh (rebuild incremental)
dbt run --full-refresh
*/

-- --- 3. BÀI TẬP (EXERCISE) ---
/*
BÀI 1: Setup dbt project:
       - dbt init ecommerce_analytics
       - Configure connection to your database
       - Create staging models for 3 tables

BÀI 2: Implement incremental model:
       - Model chỉ load orders từ ngày hôm nay
       - Handle late-arriving data

BÀI 3: Create macro để generate date spine:
       - Generate all dates from start to end
       - Useful for filling gaps in timeseries

BÀI 4: Build metrics layer:
       - Define revenue, order_count metrics
       - Use dbt-metrics hoặc MetricFlow
*/
