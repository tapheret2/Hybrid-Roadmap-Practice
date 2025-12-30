"""
================================================================
SE INTERN - BACKEND: PYTHON + FASTAPI
================================================================

Cài đặt: pip install fastapi uvicorn
Chạy: uvicorn lesson_python_fastapi:app --reload
Docs: http://localhost:8000/docs (Swagger UI tự động)
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. FastAPI: Framework Python hiện đại, nhanh, tự động tạo docs
2. Pydantic: Validation data với type hints
3. Path Parameters: /users/{id} → lấy id từ URL
4. Query Parameters: /users?skip=0&limit=10 → params tùy chọn
5. Request Body: Dữ liệu JSON gửi từ client
6. Dependency Injection: Tái sử dụng logic (auth, database)
"""

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, EmailStr
from typing import Optional, List

app = FastAPI(
    title="Learning API",
    description="API để học FastAPI basics",
    version="1.0.0"
)

# --- 2. CODE MẪU (CODE SAMPLE) ---

# Pydantic Models (Schema/DTO)
class UserCreate(BaseModel):
    name: str
    email: EmailStr  # Tự động validate email format
    age: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int]

# In-memory database
users_db: List[dict] = [
    {"id": 1, "name": "An", "email": "an@example.com", "age": 22},
    {"id": 2, "name": "Bình", "email": "binh@example.com", "age": 25},
]

# GET - Lấy tất cả users với pagination
@app.get("/api/users", response_model=List[UserResponse])
def get_users(
    skip: int = Query(default=0, ge=0, description="Số bản ghi bỏ qua"),
    limit: int = Query(default=10, le=100, description="Số bản ghi tối đa")
):
    return users_db[skip:skip + limit]

# GET - Lấy user theo ID
@app.get("/api/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int = Path(..., gt=0, description="User ID phải lớn hơn 0")
):
    user = next((u for u in users_db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# POST - Tạo user mới
@app.post("/api/users", response_model=UserResponse, status_code=201)
def create_user(user: UserCreate):
    new_user = {
        "id": len(users_db) + 1,
        **user.model_dump()  # Convert Pydantic model → dict
    }
    users_db.append(new_user)
    return new_user

# PUT - Cập nhật user
@app.put("/api/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user_update: UserCreate):
    for i, user in enumerate(users_db):
        if user["id"] == user_id:
            users_db[i] = {"id": user_id, **user_update.model_dump()}
            return users_db[i]
    raise HTTPException(status_code=404, detail="User not found")

# DELETE - Xóa user
@app.delete("/api/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    global users_db
    users_db = [u for u in users_db if u["id"] != user_id]
    return None

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Thêm endpoint GET /api/users/search?name=xxx để tìm theo tên
       Gợi ý: Dùng Query parameter

BÀI 2: Tạo model Product với fields: name, price, category, in_stock
       Tạo CRUD endpoints cho /api/products

BÀI 3: Thêm validation:
       - price phải > 0
       - category phải là một trong: ["electronics", "clothing", "food"]
       Gợi ý: Dùng Field() từ pydantic hoặc Literal

BÀI 4 (Nâng cao): Tạo dependency để kiểm tra API key từ header
       Áp dụng cho tất cả routes với Depends()
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
