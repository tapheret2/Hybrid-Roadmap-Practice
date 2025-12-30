"""
================================================================
DS INTERN - BÀI 2: NUMPY FUNDAMENTALS
================================================================

Cài đặt: pip install numpy
NumPy là nền tảng của hầu hết thư viện DS/ML trong Python
"""

import numpy as np

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. ndarray: Mảng n chiều, nhanh hơn list nhiều lần
2. Vectorization: Thực hiện operations trên toàn bộ array thay vì loop
3. Broadcasting: Tự động mở rộng dimensions khi tính toán
4. Axis: axis=0 (theo cột/row), axis=1 (theo hàng/column)
5. Shape: Kích thước của array (rows, columns, ...)
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# Tạo Array
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Array Generation
zeros = np.zeros((3, 4))           # Ma trận 3x4 toàn số 0
ones = np.ones((2, 3))             # Ma trận 2x3 toàn số 1
eye = np.eye(3)                    # Ma trận đơn vị 3x3
arange = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1]
random_arr = np.random.rand(3, 3)  # Ma trận 3x3 ngẫu nhiên (0-1)
randint_arr = np.random.randint(0, 100, (3, 3))  # Random integers

# Shape & Reshape
print(f"Shape: {arr2d.shape}")     # (2, 3)
reshaped = arr2d.reshape(3, 2)     # Đổi thành 3x2
flattened = arr2d.flatten()        # Chuyển thành 1D

# Indexing & Slicing
print(arr2d[0, 1])                 # Phần tử hàng 0, cột 1
print(arr2d[:, 1])                 # Toàn bộ cột 1
print(arr2d[1, :])                 # Toàn bộ hàng 1
print(arr2d[0:2, 1:3])             # Slice 2D

# Boolean Indexing (RẤT QUAN TRỌNG)
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3                     # [False, False, False, True, True, True]
filtered = arr[mask]               # [4, 5, 6]
filtered2 = arr[arr % 2 == 0]      # [2, 4, 6]

# Vectorization (Thay thế Loop)
data = np.array([1, 2, 3, 4, 5])
squared = data ** 2                # [1, 4, 9, 16, 25]
doubled = data * 2                 # [2, 4, 6, 8, 10]

# Aggregate Functions
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Sum: {matrix.sum()}")      # 21
print(f"Mean: {matrix.mean()}")    # 3.5
print(f"Std: {matrix.std()}")      # Standard deviation
print(f"Sum by row: {matrix.sum(axis=1)}")    # [6, 15]
print(f"Sum by col: {matrix.sum(axis=0)}")    # [5, 7, 9]
print(f"Max: {matrix.max()}, Argmax: {matrix.argmax()}")

# Broadcasting
a = np.array([[1], [2], [3]])      # Shape: (3, 1)
b = np.array([1, 2, 3])            # Shape: (3,)
result = a + b                     # Shape: (3, 3) - Tự động mở rộng

# Matrix Operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
element_mult = A * B               # Element-wise
matrix_mult = A @ B                # Matrix multiplication (dot product)
transpose = A.T                    # Chuyển vị

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Tạo array 10 số ngẫu nhiên từ 0-100, tìm:
       - Giá trị lớn nhất, nhỏ nhất
       - Trung bình, độ lệch chuẩn
       - Các số lớn hơn trung bình
"""
def analyze_random_array():
    arr = np.random.randint(0, 100, 10)
    # Viết code phân tích tại đây
    pass

"""
BÀI 2: Cho 2 ma trận A và B (3x3), tính:
       - Tổng, hiệu, tích element-wise
       - Tích ma trận (dot product)
       - Transpose của A
"""
def matrix_operations():
    A = np.random.randint(1, 10, (3, 3))
    B = np.random.randint(1, 10, (3, 3))
    # Viết code tại đây
    pass

"""
BÀI 3: Normalize một array về khoảng [0, 1]
       Công thức: (x - min) / (max - min)
       Input: np.array([10, 20, 30, 40, 50])
       Output: array([0., 0.25, 0.5, 0.75, 1.])
"""
def normalize_array(arr):
    # Viết code tại đây (KHÔNG dùng loop)
    pass

# --- TEST ---
if __name__ == "__main__":
    print("=== Hoàn thành các bài tập NumPy ===")
    # analyze_random_array()
    # matrix_operations()
    # test = np.array([10, 20, 30, 40, 50])
    # print(normalize_array(test))
