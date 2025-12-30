"""
================================================================
SE JUNIOR - BACKEND ADVANCED: AUTH, GRAPHQL, ORM
================================================================

Cài đặt: pip install fastapi sqlalchemy pyjwt bcrypt strawberry-graphql
"""

from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import bcrypt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Authentication:
   - JWT: Stateless tokens, scalable
   - OAuth 2.0: Third-party login (Google, GitHub)
   - Sessions: Server-side, more secure but stateful

2. Authorization:
   - RBAC: Role-based access control
   - ABAC: Attribute-based access control
   - Permissions: Fine-grained access

3. Security Best Practices:
   - Password hashing (bcrypt, argon2)
   - HTTPS everywhere
   - CORS configuration
   - Rate limiting
   - Input validation

4. ORM (Object-Relational Mapping):
   - SQLAlchemy (Python), Prisma (Node.js)
   - Models → Database tables
   - Relationships: One-to-Many, Many-to-Many
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ========== JWT AUTHENTICATION ==========

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return TokenData(username=username)
    except jwt.PyJWTError:
        return None

# ========== PASSWORD HASHING ==========

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ========== SQLALCHEMY ORM ==========

Base = declarative_base()
engine = create_engine("sqlite:///./app.db")
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True)
    hashed_password = Column(String(100))
    role = Column(String(20), default="user")
    
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200))
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey("users.id"))
    
    author = relationship("User", back_populates="posts")

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========== API ENDPOINTS ==========

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    
    class Config:
        from_attributes = True

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials"
        )
    
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_db)):
    token_data = verify_token(token)
    if token_data is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ========== ROLE-BASED ACCESS CONTROL ==========

def require_role(required_role: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

@app.get("/admin/users")
def list_users(
    admin: User = Depends(require_role("admin")),
    db = Depends(get_db)
):
    return db.query(User).all()

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Implement refresh token:
       - Access token: 15 min
       - Refresh token: 7 days
       - Endpoint: POST /refresh

BÀI 2: Add OAuth2 Social Login:
       - Login with Google
       - Create/link user account

BÀI 3: Build CRUD cho Posts với authorization:
       - User only edit/delete own posts
       - Admin can edit/delete any

BÀI 4: Implement rate limiting:
       - 100 requests per minute per IP
       - Use slowapi or custom middleware
"""

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables created!")
    print("Run: uvicorn lesson_backend_advanced:app --reload")
