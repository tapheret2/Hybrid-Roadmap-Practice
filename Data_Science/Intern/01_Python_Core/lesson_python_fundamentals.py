"""
================================================================
DS INTERN - LESSON 1: PYTHON FUNDAMENTALS
================================================================
"""

# --- 1. THEORY ---
"""
1. Data Types:
   - Numbers: int, float
   - Strings: str
   - Boolean: bool
   - Collections: list, tuple, dict, set

2. Control Flow:
   - if/elif/else
   - for loops
   - while loops
   - break, continue

3. Functions:
   - def keyword
   - Parameters and arguments
   - Return values
   - Lambda functions

4. OOP Basics:
   - Classes and objects
   - __init__ method
   - Instance and class attributes
   - Methods
"""

# --- 2. CODE SAMPLE ---

# ========== DATA TYPES ==========
name = "John"                  # str
age = 25                       # int
price = 99.99                  # float
is_active = True               # bool
hobbies = ["coding", "music"]  # list
point = (10, 20)               # tuple
person = {"name": "John"}      # dict
unique_ids = {1, 2, 3}         # set

# ========== LIST OPERATIONS ==========
numbers = [1, 2, 3, 4, 5]

# Append and extend
numbers.append(6)              # [1, 2, 3, 4, 5, 6]
numbers.extend([7, 8])         # [1, 2, 3, 4, 5, 6, 7, 8]

# Slicing
first_three = numbers[:3]      # [1, 2, 3]
last_two = numbers[-2:]        # [7, 8]
reversed_list = numbers[::-1]  # [8, 7, 6, 5, 4, 3, 2, 1]

# ========== DICTIONARY OPERATIONS ==========
student = {
    "name": "Alice",
    "age": 20,
    "grades": [85, 90, 88]
}

# Access
print(student["name"])                     # Alice
print(student.get("email", "N/A"))         # N/A (default)

# Update
student["email"] = "alice@example.com"
student.update({"age": 21, "city": "NYC"})

# Iterate
for key, value in student.items():
    print(f"{key}: {value}")

# ========== COMPREHENSIONS ==========
# List comprehension
squares = [x**2 for x in range(1, 6)]      # [1, 4, 9, 16, 25]
evens = [x for x in range(10) if x % 2 == 0]

# Dict comprehension
square_dict = {x: x**2 for x in range(1, 6)}  # {1: 1, 2: 4, ...}

# ========== FUNCTIONS ==========
def greet(name, greeting="Hello"):
    """Greet someone with a message"""
    return f"{greeting}, {name}!"

def calculate_stats(*numbers):
    """Calculate sum and average of numbers"""
    total = sum(numbers)
    avg = total / len(numbers) if numbers else 0
    return {"sum": total, "average": avg}

# Lambda functions
double = lambda x: x * 2
add = lambda a, b: a + b

# ========== CLASSES ==========
class BankAccount:
    """Simple bank account class"""
    
    interest_rate = 0.02  # Class attribute
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        self.transactions = []
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.transactions.append(f"+${amount}")
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.transactions.append(f"-${amount}")
            return True
        return False
    
    def get_balance(self):
        return self.balance
    
    def __str__(self):
        return f"Account({self.owner}, ${self.balance})"
    
    def __repr__(self):
        return f"BankAccount('{self.owner}', {self.balance})"

# ========== ERROR HANDLING ==========
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except TypeError:
        print("Error: Invalid types")
        return None
    else:
        return result
    finally:
        print("Division attempted")

# ========== FILE HANDLING ==========
def read_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)

# --- 3. EXERCISES ---
"""
EXERCISE 1: Create a ShoppingCart class
           - add_item(name, price, quantity)
           - remove_item(name)
           - get_total()
           - apply_discount(percent)

EXERCISE 2: Implement a number guessing game
           - Generate random number 1-100
           - User guesses until correct
           - Show "too high" or "too low" hints
           - Count attempts

EXERCISE 3: File operations
           - Read a CSV file manually
           - Parse into list of dictionaries
           - Filter and save results

EXERCISE 4: Decorator function
           - Create @timer decorator
           - Create @retry decorator
           - Apply to sample functions
"""

# === TEST ===
if __name__ == "__main__":
    print("=== Python Fundamentals ===")
    print(greet("World"))
    print(calculate_stats(1, 2, 3, 4, 5))
    
    account = BankAccount("John", 100)
    account.deposit(50)
    account.withdraw(30)
    print(account)
