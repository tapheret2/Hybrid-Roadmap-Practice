"""
================================================================
DS INTERN - BÀI 3: PANDAS MASTERY
================================================================

Cài đặt: pip install pandas
Pandas là thư viện chính để xử lý dữ liệu dạng bảng
"""

import pandas as pd
import numpy as np

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Series: Mảng 1 chiều với index
2. DataFrame: Bảng 2 chiều (như Excel/SQL table)
3. Index: Nhãn của rows, có thể là số hoặc string
4. Vectorization: Pandas cũng dùng vectorization như NumPy
5. Method Chaining: df.method1().method2().method3()
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# Tạo DataFrame
df = pd.DataFrame({
    'name': ['An', 'Bình', 'Chi', 'Dũng', 'Hoa'],
    'age': [22, 25, 23, 30, 28],
    'salary': [5000, 6000, 4500, 8000, 7000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
})

# Đọc/Ghi file
# df = pd.read_csv('data.csv')
# df.to_csv('output.csv', index=False)
# df = pd.read_excel('data.xlsx')

# Xem tổng quan dữ liệu
print(df.head())              # 5 dòng đầu
print(df.tail(3))             # 3 dòng cuối
print(df.info())              # Thông tin cột, dtype
print(df.describe())          # Thống kê cơ bản
print(df.shape)               # (rows, cols)
print(df.columns.tolist())    # Danh sách cột

# Selection (Chọn dữ liệu)
print(df['name'])             # Chọn 1 cột → Series
print(df[['name', 'age']])    # Chọn nhiều cột → DataFrame
print(df.loc[0])              # Chọn theo label (index)
print(df.iloc[0:3])           # Chọn theo vị trí số
print(df.loc[df['age'] > 25]) # Chọn theo điều kiện

# Filtering (Lọc dữ liệu)
it_staff = df[df['department'] == 'IT']
high_salary = df[df['salary'] > 5000]
combined = df[(df['age'] > 23) & (df['salary'] > 5000)]

# Creating/Modifying Columns
df['bonus'] = df['salary'] * 0.1
df['full_income'] = df['salary'] + df['bonus']
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 26 else 'Senior')

# Handling Missing Values
df_with_nan = df.copy()
df_with_nan.loc[0, 'salary'] = np.nan
print(df_with_nan.isna().sum())         # Đếm NaN theo cột
df_filled = df_with_nan.fillna(0)       # Điền NaN bằng 0
df_filled2 = df_with_nan['salary'].fillna(df_with_nan['salary'].mean())  # Điền bằng mean
df_dropped = df_with_nan.dropna()       # Xóa rows có NaN

# GroupBy (QUAN TRỌNG)
grouped = df.groupby('department')
print(grouped['salary'].mean())         # Lương TB theo phòng
print(grouped.agg({
    'salary': ['mean', 'max', 'min'],
    'age': 'mean'
}))

# Sorting
df_sorted = df.sort_values('salary', ascending=False)
df_sorted_multi = df.sort_values(['department', 'salary'], ascending=[True, False])

# Merge/Join
df2 = pd.DataFrame({
    'department': ['IT', 'HR', 'Finance'],
    'budget': [100000, 50000, 80000]
})
merged = pd.merge(df, df2, on='department', how='left')

# Pivot Table
pivot = df.pivot_table(
    values='salary',
    index='department',
    aggfunc=['mean', 'count']
)

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Cho DataFrame sales với cột: date, product, quantity, price
       Tính tổng doanh thu (quantity * price) theo từng product
"""
def calculate_revenue():
    sales = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'quantity': [10, 5, 8, 3, 6, 12, 4, 7, 9, 5],
        'price': [100, 200, 100, 150, 200, 100, 150, 200, 100, 150]
    })
    # Viết code tại đây
    pass

"""
BÀI 2: Xử lý dữ liệu có missing values
       - Điền missing 'age' bằng trung bình theo 'department'
       - Xóa rows có 'salary' bị NaN
"""
def handle_missing():
    data = pd.DataFrame({
        'name': ['A', 'B', 'C', 'D', 'E'],
        'age': [22, np.nan, 25, np.nan, 30],
        'salary': [5000, 6000, np.nan, 7000, 8000],
        'department': ['IT', 'IT', 'HR', 'IT', 'HR']
    })
    # Viết code tại đây
    pass

"""
BÀI 3: Tạo báo cáo tổng hợp từ df:
       - Số nhân viên mỗi phòng
       - Lương trung bình, min, max mỗi phòng
       - Phòng có lương TB cao nhất
"""
def department_report(df):
    # Viết code tại đây
    pass

# --- TEST ---
if __name__ == "__main__":
    print("=== DataFrame mẫu ===")
    print(df)
    print("\n=== Hoàn thành các bài tập Pandas ===")
