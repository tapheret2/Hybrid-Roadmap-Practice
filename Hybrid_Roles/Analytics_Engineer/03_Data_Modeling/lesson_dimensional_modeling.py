"""
================================================================
ANALYTICS ENGINEER - MODULE 3: DATA MODELING & KIMBALL
================================================================

Data Modeling là kỹ năng cốt lõi của Analytics Engineer
Thiết kế data models tối ưu cho analytics queries
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Kimball Methodology:
   - Dimensional Modeling
   - Star Schema: Facts + Dimensions
   - Designed for query performance
   - Business process-centric

2. Data Vault:
   - Hub: Business keys
   - Link: Relationships
   - Satellite: Descriptive data
   - Good for enterprise, audit trails

3. One Big Table (OBT):
   - Denormalized single table
   - Simple, fast for BI tools
   - Trade-off: Storage vs Query speed

4. Key Concepts:
   - Grain: Level of detail in fact table
   - Conformed Dimensions: Shared across facts
   - Junk Dimension: Low-cardinality flags
   - Slowly Changing Dimension (SCD)

5. Naming Conventions:
   - dim_* : Dimension tables
   - fct_* : Fact tables
   - stg_* : Staging models
   - int_* : Intermediate models
   - rpt_* : Report-ready tables
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== DIMENSIONAL MODELING EXAMPLE ==========

def create_date_dimension(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create a date dimension table
    Essential for any analytics project
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df = pd.DataFrame({'date_key': dates})
    df['date_key_int'] = df['date_key'].dt.strftime('%Y%m%d').astype(int)
    df['full_date'] = df['date_key'].dt.date
    df['day_of_week'] = df['date_key'].dt.day_name()
    df['day_of_week_num'] = df['date_key'].dt.dayofweek
    df['day_of_month'] = df['date_key'].dt.day
    df['day_of_year'] = df['date_key'].dt.dayofyear
    df['week_of_year'] = df['date_key'].dt.isocalendar().week
    df['month_num'] = df['date_key'].dt.month
    df['month_name'] = df['date_key'].dt.month_name()
    df['quarter'] = df['date_key'].dt.quarter
    df['year'] = df['date_key'].dt.year
    df['is_weekend'] = df['day_of_week_num'] >= 5
    df['is_month_start'] = df['date_key'].dt.is_month_start
    df['is_month_end'] = df['date_key'].dt.is_month_end
    
    # Fiscal calendar (example: FY starts in July)
    df['fiscal_year'] = df.apply(
        lambda x: x['year'] + 1 if x['month_num'] >= 7 else x['year'], 
        axis=1
    )
    df['fiscal_quarter'] = ((df['month_num'] - 7) % 12) // 3 + 1
    
    return df

def build_customer_dimension(raw_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer dimension with SCD Type 2 support
    """
    dim = raw_customers.copy()
    
    # Generate surrogate key
    dim['customer_key'] = range(1, len(dim) + 1)
    
    # Add SCD columns
    dim['effective_date'] = datetime.now()
    dim['expiration_date'] = datetime(9999, 12, 31)
    dim['is_current'] = True
    
    # Derived attributes
    dim['customer_segment'] = dim['total_orders'].apply(
        lambda x: 'VIP' if x >= 10 else 'Regular' if x >= 3 else 'New'
    )
    
    return dim[['customer_key', 'customer_id', 'customer_name', 'email', 
                'city', 'country', 'customer_segment', 
                'effective_date', 'expiration_date', 'is_current']]

def build_fact_table(
    orders: pd.DataFrame, 
    dim_customer: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_date: pd.DataFrame
) -> pd.DataFrame:
    """
    Build fact table by joining with dimensions
    """
    fact = orders.copy()
    
    # Join with dimensions to get surrogate keys
    fact = fact.merge(
        dim_customer[['customer_id', 'customer_key']], 
        on='customer_id', 
        how='left'
    )
    fact = fact.merge(
        dim_product[['product_id', 'product_key']], 
        on='product_id', 
        how='left'
    )
    
    # Convert order_date to date_key
    fact['date_key'] = fact['order_date'].dt.strftime('%Y%m%d').astype(int)
    
    # Calculate measures
    fact['line_total'] = fact['quantity'] * fact['unit_price']
    fact['discount_amount'] = fact['line_total'] * fact['discount_pct']
    fact['net_amount'] = fact['line_total'] - fact['discount_amount']
    
    # Select final columns
    return fact[[
        'order_id', 'date_key', 'customer_key', 'product_key',
        'quantity', 'unit_price', 'discount_pct',
        'line_total', 'discount_amount', 'net_amount'
    ]]

# ========== ONE BIG TABLE (OBT) ==========

def build_one_big_table(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame
) -> pd.DataFrame:
    """
    Build OBT for simple BI needs
    Pros: Fast queries, simple
    Cons: Redundant data, large storage
    """
    obt = orders.merge(customers, on='customer_id', how='left', suffixes=('', '_customer'))
    obt = obt.merge(products, on='product_id', how='left', suffixes=('', '_product'))
    
    # Add calculated fields
    obt['order_month'] = obt['order_date'].dt.to_period('M')
    obt['order_year'] = obt['order_date'].dt.year
    obt['line_total'] = obt['quantity'] * obt['unit_price']
    
    # Customer lifetime metrics
    customer_totals = obt.groupby('customer_id').agg({
        'line_total': 'sum',
        'order_id': 'nunique'
    }).rename(columns={
        'line_total': 'customer_ltv',
        'order_id': 'customer_order_count'
    })
    
    obt = obt.merge(customer_totals, on='customer_id', how='left')
    
    return obt

# ========== DATA MODEL DOCUMENTATION ==========

def generate_model_documentation(model_name: str, columns: List[Dict]) -> str:
    """
    Generate documentation for a data model
    Important for data discovery
    """
    doc = f"""
# {model_name}

## Description
[Add description here]

## Grain
[Describe what one row represents]

## Columns

| Column | Type | Description |
|--------|------|-------------|
"""
    for col in columns:
        doc += f"| {col['name']} | {col['type']} | {col['description']} |\n"
    
    doc += """
## Dependencies
- [List upstream models]

## Usage
```sql
SELECT * FROM {model_name} WHERE date_key >= 20240101
```
"""
    return doc

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Design Star Schema cho E-commerce:
       - Fact: Sales (grain: order line level)
       - Dimensions: Date, Customer, Product, Store, Promotion
       - Draw ERD diagram

BÀI 2: Implement SCD Type 2:
       - Track customer address changes
       - Support point-in-time queries
       - Ensure historical accuracy

BÀI 3: Build semantic layer:
       - Define business metrics (Revenue, Margin, etc.)
       - Create consistent calculations
       - Document for business users

BÀI 4: Compare modeling approaches:
       - Build same report with Star Schema vs OBT
       - Measure query performance
       - Analyze storage requirements
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== Data Modeling Demo ===\n")
    
    # Create date dimension
    dim_date = create_date_dimension('2024-01-01', '2024-12-31')
    print(f"Date Dimension: {len(dim_date)} rows")
    print(dim_date.head(3))
    
    # Generate model documentation
    columns = [
        {'name': 'order_id', 'type': 'INT', 'description': 'Primary key'},
        {'name': 'customer_key', 'type': 'INT', 'description': 'FK to dim_customer'},
        {'name': 'net_amount', 'type': 'DECIMAL', 'description': 'Total after discount'},
    ]
    print("\n" + generate_model_documentation('fct_orders', columns))
