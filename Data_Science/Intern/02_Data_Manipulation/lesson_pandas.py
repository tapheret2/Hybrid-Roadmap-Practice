"""
================================================================
DS INTERN - LESSON 3: PANDAS MASTERY
================================================================

Pandas = Panel Data
The most important library for data manipulation

Install: pip install pandas
"""

import pandas as pd
import numpy as np

# --- 1. THEORY ---
"""
1. Core Objects:
   - Series: 1D labeled array
   - DataFrame: 2D labeled table (like Excel)

2. Key Operations:
   - Selection: loc (label), iloc (integer)
   - Filtering: Boolean indexing
   - GroupBy: Split-Apply-Combine
   - Merge/Join: Combine DataFrames

3. Data Cleaning:
   - Missing values: dropna, fillna
   - Duplicates: drop_duplicates
   - Type conversion: astype

4. Performance Tips:
   - Use vectorized operations
   - Avoid iterating with for loops
   - Use categories for repeated strings
"""

# --- 2. CODE SAMPLE ---

# ========== CREATE DATAFRAME ==========
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
    'salary': [70000, 85000, 90000, 75000, 80000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing']
})

print("DataFrame:\n", df)
print("\nInfo:")
print(df.info())
print("\nDescribe:\n", df.describe())

# ========== SELECTION ==========
# Column selection
print(df['name'])                           # Single column (Series)
print(df[['name', 'salary']])               # Multiple columns (DataFrame)

# Row selection
print(df.loc[0])                            # By label
print(df.iloc[0])                           # By integer position
print(df.loc[0:2, 'name':'city'])           # Slice by label
print(df.iloc[0:3, 0:2])                    # Slice by position

# ========== FILTERING ==========
# Single condition
high_salary = df[df['salary'] > 80000]

# Multiple conditions
nyc_engineers = df[(df['city'] == 'NYC') & (df['department'] == 'Engineering')]

# isin
la_chi = df[df['city'].isin(['LA', 'Chicago'])]

# Query method (cleaner syntax)
result = df.query("salary > 75000 and city == 'NYC'")

# ========== CREATE/MODIFY COLUMNS ==========
df['bonus'] = df['salary'] * 0.1
df['full_name'] = df['name'] + ' (' + df['department'] + ')'
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 100], labels=['Young', 'Mid', 'Senior'])

# Apply function
df['salary_level'] = df['salary'].apply(lambda x: 'High' if x > 80000 else 'Normal')

# ========== HANDLING MISSING VALUES ==========
df_with_na = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': ['x', None, 'z', 'w']
})

print("Missing values:\n", df_with_na.isnull().sum())

# Remove rows with any missing
df_clean = df_with_na.dropna()

# Remove rows if specific column is missing
df_clean = df_with_na.dropna(subset=['A'])

# Fill missing values
df_filled = df_with_na.fillna({
    'A': df_with_na['A'].mean(),
    'B': 0,
    'C': 'Unknown'
})

# Forward/backward fill
df_with_na['A'].fillna(method='ffill')      # Forward fill
df_with_na['A'].fillna(method='bfill')      # Backward fill

# ========== GROUPBY ==========
# Basic groupby
by_city = df.groupby('city')['salary'].mean()
print("Average salary by city:\n", by_city)

# Multiple aggregations
summary = df.groupby('city').agg({
    'salary': ['mean', 'min', 'max', 'count'],
    'age': 'mean'
})
print("Summary:\n", summary)

# Named aggregations
summary = df.groupby('city').agg(
    avg_salary=('salary', 'mean'),
    employee_count=('name', 'count'),
    avg_age=('age', 'mean')
)

# Transform (returns same shape)
df['salary_zscore'] = df.groupby('city')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ========== SORTING ==========
df_sorted = df.sort_values('salary', ascending=False)
df_sorted = df.sort_values(['city', 'salary'], ascending=[True, False])

# ========== MERGE/JOIN ==========
departments = pd.DataFrame({
    'department': ['Engineering', 'Marketing', 'HR', 'Finance'],
    'budget': [1000000, 500000, 300000, 200000],
    'head': ['John', 'Jane', 'Mike', 'Sarah']
})

# Inner join
merged = pd.merge(df, departments, on='department', how='inner')

# Left join (keep all employees)
merged = pd.merge(df, departments, on='department', how='left')

# ========== PIVOT TABLE ==========
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='department',
    aggfunc='mean',
    fill_value=0
)
print("Pivot table:\n", pivot)

# --- 3. EXERCISES ---
"""
EXERCISE 1: EDA on a dataset
           - Load CSV file
           - Check shape, dtypes, missing values
           - Summary statistics for numeric columns
           - Value counts for categorical columns

EXERCISE 2: Data cleaning pipeline
           - Handle missing values appropriately
           - Remove duplicates
           - Fix data types
           - Normalize/standardize values

EXERCISE 3: Complex groupby
           - Calculate running total within groups
           - Rank items within groups
           - Get top N per group

EXERCISE 4: Merge multiple tables
           - Users, Orders, Products tables
           - Calculate total spend per user
           - Find most popular products
"""

if __name__ == "__main__":
    print("=== Pandas Mastery ===")
    print(df.head())

# --- EXERCISE 1 SOLUTION: EDA on a dataset ---
"""
Vì không có file CSV sẵn, ta sẽ tạo sample data trong code
Bạn có thể thay thế bằng: df_eda = pd.read_csv('your_file.csv')
"""

def exercise_1_eda():
    print("\n" + "="*60)
    print("EXERCISE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # 1. Tạo sample dataset (thay thế bằng pd.read_csv() nếu có file)
    df_eda = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 
                 'Grace', 'Henry', 'Ivy', 'Jack', 'Alice', 'Kate', 
                 'Leo', 'Mike', 'Nancy'],
        'age': [25, 30, 35, 28, 32, None, 29, 31, 27, 33, 25, 26, 34, None, 30],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC', 'Chicago', 
                 'LA', 'NYC', 'Chicago', 'NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'salary': [70000, 85000, 90000, 75000, 80000, 95000, None, 
                   82000, 72000, 78000, 70000, 68000, 88000, 73000, 81000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 
                       'Marketing', 'Engineering', 'HR', 'Finance', 
                       'Engineering', 'Marketing', 'Engineering', 'HR', 
                       'Finance', 'Engineering', 'Marketing'],
        'start_date': ['2020-01-15', '2019-06-20', '2018-03-10', '2021-09-01',
                       '2020-11-22', '2017-05-18', '2022-01-05', '2019-08-12',
                       '2021-03-25', None, '2020-01-15', '2022-06-30',
                       '2018-10-14', '2020-07-08', '2019-12-03']
    })
    
    # 2. Check shape (số hàng, số cột)
    print("\n1. SHAPE:")
    print(f"   Rows: {df_eda.shape[0]}, Columns: {df_eda.shape[1]}")
    
    # 3. Check dtypes (kiểu dữ liệu của từng cột)
    print("\n2. DATA TYPES:")
    print(df_eda.dtypes.to_string(header=False))
    
    # 4. Check missing values
    print("\n3. MISSING VALUES:")
    missing = df_eda.isnull().sum()
    missing_pct = (df_eda.isnull().sum() / len(df_eda) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].to_string())
    
    # 5. Summary statistics cho numeric columns
    print("\n4. SUMMARY STATISTICS (Numeric Columns):")
    print(df_eda.describe().round(2).to_string())
    
    # 6. Value counts cho categorical columns
    print("\n5. VALUE COUNTS (Categorical Columns):")
    
    categorical_cols = df_eda.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'name':  # Bỏ qua cột name vì mỗi người khác nhau
            print(f"\n   {col.upper()}:")
            print(df_eda[col].value_counts().to_string())
    
    # 7. Check duplicates
    print("\n6. DUPLICATES:")
    duplicates = df_eda.duplicated().sum()
    print(f"   Total duplicate rows: {duplicates}")
    if duplicates > 0:
        print("   Duplicate rows:")
        print(df_eda[df_eda.duplicated(keep=False)])
    
    # 8. Additional info
    print("\n7. MEMORY USAGE:")
    print(f"   Total: {df_eda.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    return df_eda

# Chạy Exercise 1
# df_result = exercise_1_eda()
