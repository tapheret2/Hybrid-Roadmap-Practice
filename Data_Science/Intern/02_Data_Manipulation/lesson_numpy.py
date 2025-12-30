"""
================================================================
DS INTERN - LESSON 2: NUMPY FUNDAMENTALS
================================================================

NumPy = Numerical Python
Foundation of scientific computing in Python

Install: pip install numpy
"""

import numpy as np

# --- 1. THEORY ---
"""
1. ndarray: N-dimensional array
   - Homogeneous: All elements same type
   - Fixed size: Defined at creation
   - Fast: Vectorized operations

2. Vectorization:
   - Operations on entire arrays at once
   - No explicit loops needed
   - 10-100x faster than Python loops

3. Broadcasting:
   - Arrays with different shapes can be compatible
   - Smaller array is "broadcast" across larger array
   - Rules: Dimensions must be equal or one must be 1

4. Key Concepts:
   - Shape: Dimensions (rows, columns)
   - Axis: Direction of operation (0=rows, 1=columns)
   - dtype: Data type (int32, float64, etc.)
"""

# --- 2. CODE SAMPLE ---

# ========== ARRAY CREATION ==========
arr1 = np.array([1, 2, 3, 4, 5])           # From list
arr2 = np.zeros((3, 4))                     # 3x4 zeros
arr3 = np.ones((2, 3))                      # 2x3 ones
arr4 = np.full((2, 2), 7)                   # 2x2 filled with 7
arr5 = np.arange(0, 10, 2)                  # [0, 2, 4, 6, 8]
arr6 = np.linspace(0, 1, 5)                 # 5 numbers from 0 to 1
arr7 = np.eye(3)                            # 3x3 identity matrix
arr8 = np.random.rand(3, 3)                 # Random 0-1
arr9 = np.random.randn(3, 3)                # Random normal distribution
arr10 = np.random.randint(0, 10, (3, 3))    # Random integers

print("Shape:", arr8.shape)
print("Dimensions:", arr8.ndim)
print("Data type:", arr8.dtype)
print("Size:", arr8.size)

# ========== INDEXING & SLICING ==========
matrix = np.arange(1, 13).reshape(3, 4)
print("Matrix:\n", matrix)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print("Element [1,2]:", matrix[1, 2])       # 7
print("Row 1:", matrix[1, :])               # [5, 6, 7, 8]
print("Column 2:", matrix[:, 2])            # [3, 7, 11]
print("Submatrix:\n", matrix[0:2, 1:3])     # [[2, 3], [6, 7]]

# Boolean indexing
arr = np.array([1, 2, 3, 4, 5])
print("Greater than 2:", arr[arr > 2])      # [3, 4, 5]

# Fancy indexing
indices = [0, 2, 4]
print("Selected indices:", arr[indices])    # [1, 3, 5]

# ========== VECTORIZED OPERATIONS ==========
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("Addition:", a + b)                   # [11, 22, 33, 44]
print("Multiplication:", a * b)             # [10, 40, 90, 160]
print("Power:", a ** 2)                     # [1, 4, 9, 16]
print("Square root:", np.sqrt(a))           # [1.0, 1.41, 1.73, 2.0]
print("Exponential:", np.exp(a))            # [2.72, 7.39, 20.09, 54.60]
print("Log:", np.log(a))                    # [0, 0.69, 1.10, 1.39]

# ========== BROADCASTING ==========
# Array + Scalar
arr = np.array([1, 2, 3])
print("arr + 10:", arr + 10)                # [11, 12, 13]

# 2D + 1D
matrix = np.ones((3, 3))
row = np.array([1, 2, 3])
print("Matrix + row:\n", matrix + row)
# [[2. 3. 4.]
#  [2. 3. 4.]
#  [2. 3. 4.]]

# ========== AGGREGATE FUNCTIONS ==========
data = np.random.randn(5, 4)

print("Sum:", np.sum(data))
print("Sum by column:", np.sum(data, axis=0))  # Sum each column
print("Sum by row:", np.sum(data, axis=1))     # Sum each row
print("Mean:", np.mean(data))
print("Std:", np.std(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Argmax:", np.argmax(data))              # Index of max

# ========== MATRIX OPERATIONS ==========
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix multiply:\n", A @ B)             # or np.dot(A, B)
print("Transpose:\n", A.T)
print("Inverse:\n", np.linalg.inv(A))
print("Determinant:", np.linalg.det(A))

# ========== RESHAPING ==========
arr = np.arange(12)
print("Original:", arr)
print("Reshape 3x4:\n", arr.reshape(3, 4))
print("Reshape 2x6:\n", arr.reshape(2, -1))    # -1 = auto-calculate
print("Flatten:", arr.reshape(3, 4).flatten())

# --- 3. EXERCISES ---
"""
EXERCISE 1: Array operations
           - Create 5x5 matrix with values 1-25
           - Extract diagonal elements
           - Calculate sum of anti-diagonal

EXERCISE 2: Statistics
           - Generate 1000 random samples from normal distribution
           - Calculate mean, std, percentiles
           - Count values within 1, 2, 3 standard deviations

EXERCISE 3: Image as array
           - Create 100x100 grayscale "image" (random values 0-255)
           - Apply threshold: values > 128 become 255, else 0
           - Calculate histogram of pixel values

EXERCISE 4: Matrix operations
           - Solve system of linear equations: Ax = b
           - Calculate eigenvalues and eigenvectors
           - Normalize rows to sum to 1
"""

if __name__ == "__main__":
    print("=== NumPy Fundamentals ===")
