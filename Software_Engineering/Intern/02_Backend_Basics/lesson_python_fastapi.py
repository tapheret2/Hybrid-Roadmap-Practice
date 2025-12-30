"""
================================================================
SE INTERN - LESSON 6: PYTHON + FASTAPI
================================================================

Install: pip install fastapi uvicorn pydantic
Run: uvicorn lesson_python_fastapi:app --reload
"""

from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# --- 1. THEORY ---
"""
1. FastAPI:
   - Modern, fast Python web framework
   - Automatic OpenAPI/Swagger docs
   - Type hints for validation

2. Pydantic Models:
   - Data validation with Python types
   - Automatic JSON serialization
   - Field constraints

3. Path & Query Parameters:
   - Path: /users/{user_id}
   - Query: /users?page=1&limit=10

4. HTTP Methods:
   - GET: Read
   - POST: Create
   - PUT: Full update
   - PATCH: Partial update
   - DELETE: Remove
"""

# --- 2. CODE SAMPLE ---

app = FastAPI(
    title="Product API",
    description="A simple REST API for products",
    version="1.0.0"
)

# ========== MODELS ==========

class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    stock: int = Field(default=0, ge=0)
    category: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    price: Optional[float] = Field(None, gt=0)
    stock: Optional[int] = Field(None, ge=0)
    category: Optional[str] = None

class Product(ProductBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ProductResponse(BaseModel):
    success: bool
    data: Product

class ProductListResponse(BaseModel):
    success: bool
    count: int
    data: List[Product]

# ========== IN-MEMORY DATABASE ==========

products_db: List[dict] = [
    {"id": 1, "name": "Laptop", "price": 999.99, "stock": 10, "category": "Electronics", "created_at": datetime.now()},
    {"id": 2, "name": "Phone", "price": 699.99, "stock": 25, "category": "Electronics", "created_at": datetime.now()},
    {"id": 3, "name": "Book", "price": 29.99, "stock": 100, "category": "Books", "created_at": datetime.now()},
]
next_id = 4

# ========== ENDPOINTS ==========

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to Product API", "docs": "/docs"}

@app.get("/api/products", response_model=ProductListResponse, tags=["Products"])
def get_products(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, gt=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, gt=0, description="Maximum price")
):
    """Get all products with optional filtering and pagination"""
    result = products_db.copy()
    
    if category:
        result = [p for p in result if p.get("category") == category]
    if min_price:
        result = [p for p in result if p["price"] >= min_price]
    if max_price:
        result = [p for p in result if p["price"] <= max_price]
    
    paginated = result[skip : skip + limit]
    
    return {"success": True, "count": len(paginated), "data": paginated}

@app.get("/api/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(
    product_id: int = Path(..., gt=0, description="Product ID")
):
    """Get a single product by ID"""
    product = next((p for p in products_db if p["id"] == product_id), None)
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {"success": True, "data": product}

@app.post("/api/products", response_model=ProductResponse, status_code=201, tags=["Products"])
def create_product(product: ProductCreate):
    """Create a new product"""
    global next_id
    
    new_product = {
        "id": next_id,
        **product.model_dump(),
        "created_at": datetime.now()
    }
    products_db.append(new_product)
    next_id += 1
    
    return {"success": True, "data": new_product}

@app.put("/api/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def update_product(
    product_id: int = Path(..., gt=0),
    product: ProductCreate = None
):
    """Update a product (full update)"""
    index = next((i for i, p in enumerate(products_db) if p["id"] == product_id), None)
    
    if index is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    products_db[index] = {
        "id": product_id,
        **product.model_dump(),
        "created_at": products_db[index]["created_at"]
    }
    
    return {"success": True, "data": products_db[index]}

@app.patch("/api/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def partial_update_product(
    product_id: int = Path(..., gt=0),
    product: ProductUpdate = None
):
    """Partially update a product"""
    index = next((i for i, p in enumerate(products_db) if p["id"] == product_id), None)
    
    if index is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    update_data = product.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        products_db[index][key] = value
    
    return {"success": True, "data": products_db[index]}

@app.delete("/api/products/{product_id}", tags=["Products"])
def delete_product(product_id: int = Path(..., gt=0)):
    """Delete a product"""
    index = next((i for i, p in enumerate(products_db) if p["id"] == product_id), None)
    
    if index is None:
        raise HTTPException(status_code=404, detail="Product not found")
    
    deleted = products_db.pop(index)
    
    return {"success": True, "message": f"Product '{deleted['name']}' deleted"}

# --- 3. EXERCISES ---
"""
EXERCISE 1: Add search endpoint
           - GET /api/products/search?q=laptop
           - Search in name and category
           - Case-insensitive

EXERCISE 2: Add bulk operations
           - POST /api/products/bulk (create multiple)
           - DELETE /api/products/bulk (delete by IDs list)

EXERCISE 3: Add sorting
           - Query params: ?sort_by=price&order=desc
           - Support multiple fields: ?sort_by=category,price

EXERCISE 4: Add error handling middleware
           - Custom exception handlers
           - Consistent error response format
           - Logging
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
