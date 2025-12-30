"""
================================================================
DS INTERN - BÀI 1: PYTHON FUNDAMENTALS
================================================================

Mục tiêu: Nhớ các cú pháp Python quan trọng cho Data Science
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Variables: Không cần khai báo type, Python tự suy luận
2. Data Types: int, float, str, bool, list, dict, tuple, set
3. Control Flow: if/elif/else, for, while
4. Functions: def, args, kwargs, lambda
5. OOP Basics: class, __init__, self, inheritance
6. List Comprehension: [expr for item in list if condition]
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# Variables & Types
name = "Data Scientist"
age = 22
salary = 15000.50
is_active = True
skills = ["Python", "SQL", "ML"]
info = {"name": name, "age": age}

# List Operations
numbers = [1, 2, 3, 4, 5]
numbers.append(6)           # [1, 2, 3, 4, 5, 6]
numbers.extend([7, 8])      # [1, 2, 3, 4, 5, 6, 7, 8]
sliced = numbers[2:5]       # [3, 4, 5]
reversed_list = numbers[::-1]  # Đảo ngược

# Dictionary Operations
person = {"name": "An", "age": 22}
person["email"] = "an@example.com"  # Thêm key
person.get("phone", "N/A")          # Lấy với default value
for key, value in person.items():
    print(f"{key}: {value}")

# List Comprehension (RẤT QUAN TRỌNG)
squares = [x ** 2 for x in range(1, 6)]  # [1, 4, 9, 16, 25]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# Dictionary Comprehension
word_lengths = {word: len(word) for word in ["hello", "world"]}

# Functions
def greet(name, greeting="Hello"):
    """Docstring: Mô tả hàm"""
    return f"{greeting}, {name}!"

# Lambda (Anonymous Function)
square = lambda x: x ** 2
sorted_items = sorted(skills, key=lambda x: len(x))

# Args & Kwargs
def flexible_func(*args, **kwargs):
    print("Args:", args)
    print("Kwargs:", kwargs)

# OOP Basics
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def summary(self):
        return {
            "count": len(self.data),
            "sum": sum(self.data),
            "mean": sum(self.data) / len(self.data)
        }

# Error Handling
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Cannot divide by zero"
    finally:
        print("Division attempted")

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Viết hàm nhận một list số và trả về list chỉ chứa số nguyên tố
       Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
       Output: [2, 3, 5, 7]
"""
def filter_primes(numbers):
    # Viết code tại đây
    pass

"""
BÀI 2: Viết hàm đếm tần suất xuất hiện của các từ trong một câu
       Input: "hello world hello python world world"
       Output: {"hello": 2, "world": 3, "python": 1}
"""
def word_frequency(sentence):
    # Gợi ý: Dùng dict comprehension hoặc collections.Counter
    pass

"""
BÀI 3: Tạo class Student với:
       - Attributes: name, scores (list điểm các môn)
       - Method: average() trả về điểm trung bình
       - Method: grade() trả về xếp loại (A/B/C/D/F)
"""
class Student:
    pass

# --- TEST ---
if __name__ == "__main__":
    print("=== Hoàn thành các bài tập trên để kiểm tra ===")
    # print(filter_primes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    # print(word_frequency("hello world hello python world world"))
