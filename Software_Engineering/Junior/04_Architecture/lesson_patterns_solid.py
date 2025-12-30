"""
================================================================
SE JUNIOR - ARCHITECTURE: DESIGN PATTERNS & SOLID
================================================================

Các patterns và principles quan trọng cho code maintainable
"""

from abc import ABC, abstractmethod
from typing import List, Protocol
from dataclasses import dataclass
from datetime import datetime

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. SOLID Principles:
   - S: Single Responsibility - Một class chỉ làm một việc
   - O: Open/Closed - Open for extension, closed for modification
   - L: Liskov Substitution - Subclass có thể thay thế parent
   - I: Interface Segregation - Interface nhỏ, specific
   - D: Dependency Inversion - Depend on abstractions

2. Design Patterns:
   - Creational: Factory, Singleton, Builder
   - Structural: Adapter, Decorator
   - Behavioral: Strategy, Observer, Repository

3. Clean Architecture:
   - Entities: Business logic
   - Use Cases: Application logic
   - Controllers: Interface adapters
   - Frameworks: External tools

4. Microservices:
   - Independent deployment
   - Single responsibility per service
   - API communication (REST, gRPC)
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== SOLID PRINCIPLES ==========

# S - Single Responsibility
class UserValidator:
    def validate(self, user_data: dict) -> bool:
        return bool(user_data.get('email') and user_data.get('name'))

class UserRepository:
    def save(self, user: dict) -> None:
        print(f"Saving user: {user}")

class EmailService:
    def send_welcome(self, email: str) -> None:
        print(f"Sending welcome email to {email}")

# Bad: One class doing everything
# Good: Separated concerns

# O - Open/Closed
class Discount(ABC):
    @abstractmethod
    def calculate(self, price: float) -> float:
        pass

class PercentageDiscount(Discount):
    def __init__(self, percent: float):
        self.percent = percent
    
    def calculate(self, price: float) -> float:
        return price * (1 - self.percent / 100)

class FixedDiscount(Discount):
    def __init__(self, amount: float):
        self.amount = amount
    
    def calculate(self, price: float) -> float:
        return max(0, price - self.amount)

# Adding new discount type doesn't modify existing code

# D - Dependency Inversion
class PaymentGateway(Protocol):
    def charge(self, amount: float) -> bool: ...

class StripeGateway:
    def charge(self, amount: float) -> bool:
        print(f"Charging ${amount} via Stripe")
        return True

class PayPalGateway:
    def charge(self, amount: float) -> bool:
        print(f"Charging ${amount} via PayPal")
        return True

class OrderService:
    def __init__(self, payment: PaymentGateway):
        self.payment = payment  # Depend on abstraction
    
    def checkout(self, amount: float) -> bool:
        return self.payment.charge(amount)

# ========== DESIGN PATTERNS ==========

# Factory Pattern
class NotificationFactory:
    @staticmethod
    def create(channel: str) -> 'Notification':
        if channel == "email":
            return EmailNotification()
        elif channel == "sms":
            return SMSNotification()
        elif channel == "push":
            return PushNotification()
        raise ValueError(f"Unknown channel: {channel}")

class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass

class EmailNotification(Notification):
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class SMSNotification(Notification):
    def send(self, message: str) -> None:
        print(f"SMS: {message}")

class PushNotification(Notification):
    def send(self, message: str) -> None:
        print(f"Push: {message}")

# Repository Pattern
@dataclass
class User:
    id: int
    name: str
    email: str
    created_at: datetime = None

class UserRepositoryInterface(ABC):
    @abstractmethod
    def find_by_id(self, id: int) -> User | None: pass
    
    @abstractmethod
    def find_all(self) -> List[User]: pass
    
    @abstractmethod
    def save(self, user: User) -> User: pass

class InMemoryUserRepository(UserRepositoryInterface):
    def __init__(self):
        self._users: dict[int, User] = {}
        self._next_id = 1
    
    def find_by_id(self, id: int) -> User | None:
        return self._users.get(id)
    
    def find_all(self) -> List[User]:
        return list(self._users.values())
    
    def save(self, user: User) -> User:
        if user.id is None:
            user.id = self._next_id
            self._next_id += 1
        user.created_at = datetime.now()
        self._users[user.id] = user
        return user

# Strategy Pattern
class PricingStrategy(ABC):
    @abstractmethod
    def compute(self, base_price: float) -> float:
        pass

class RegularPricing(PricingStrategy):
    def compute(self, base_price: float) -> float:
        return base_price

class PremiumPricing(PricingStrategy):
    def compute(self, base_price: float) -> float:
        return base_price * 0.8  # 20% discount

class Product:
    def __init__(self, name: str, base_price: float, strategy: PricingStrategy):
        self.name = name
        self.base_price = base_price
        self.strategy = strategy
    
    def get_price(self) -> float:
        return self.strategy.compute(self.base_price)

# ========== CLEAN ARCHITECTURE ==========

# Entities (Domain)
@dataclass
class Order:
    id: int
    user_id: int
    items: List[dict]
    total: float
    status: str = "pending"

# Use Cases (Application)
class CreateOrderUseCase:
    def __init__(self, order_repo, payment_gateway, notification_service):
        self.order_repo = order_repo
        self.payment = payment_gateway
        self.notification = notification_service
    
    def execute(self, user_id: int, items: List[dict]) -> Order:
        total = sum(item['price'] * item['quantity'] for item in items)
        
        order = Order(id=None, user_id=user_id, items=items, total=total)
        
        if self.payment.charge(total):
            order.status = "paid"
            self.order_repo.save(order)
            self.notification.send(f"Order {order.id} confirmed!")
        
        return order

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Refactor a "god class" thành nhiều SRP classes

BÀI 2: Implement Observer pattern:
       - EventEmitter class
       - Subscribe/Unsubscribe
       - Async event handling

BÀI 3: Build a simple Clean Architecture app:
       - User registration use case
       - Repository interface
       - In-memory and PostgreSQL implementations

BÀI 4: Implement Decorator pattern:
       - Base Logger
       - Add timestamp decorator
       - Add context decorator
"""

if __name__ == "__main__":
    print("=== Design Patterns Demo ===\n")
    
    # Factory
    notif = NotificationFactory.create("email")
    notif.send("Hello!")
    
    # Repository
    repo = InMemoryUserRepository()
    user = repo.save(User(id=None, name="An", email="an@test.com"))
    print(f"Saved: {user}")
    
    # Strategy
    regular = Product("Laptop", 1000, RegularPricing())
    premium = Product("Laptop", 1000, PremiumPricing())
    print(f"Regular: ${regular.get_price()}, Premium: ${premium.get_price()}")
    
    # Dependency Inversion
    stripe = StripeGateway()
    order_service = OrderService(stripe)
    order_service.checkout(99.99)
