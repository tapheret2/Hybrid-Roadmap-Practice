"""
================================================================
ML ENGINEER - MODULE 2: MODEL SERVING & DEPLOYMENT
================================================================

Mục tiêu: Đưa model từ file .pkl thành API có thể gọi được

Cài đặt: pip install fastapi uvicorn joblib scikit-learn
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Model Serialization:
   - pickle/joblib: Python native, đơn giản
   - ONNX: Cross-platform, tối ưu inference
   - SavedModel (TensorFlow), TorchScript (PyTorch)

2. Serving Patterns:
   - REST API: FastAPI, Flask (đơn giản, phổ biến)
   - gRPC: Nhanh hơn, dùng cho internal services
   - Batch: Xử lý hàng loạt, không real-time

3. Containerization:
   - Docker: Đóng gói model + dependencies
   - Reproducible environment
   - Easy scaling với Kubernetes

4. Serving Frameworks:
   - TensorFlow Serving: Production-ready cho TF models
   - TorchServe: PyTorch models
   - Seldon Core: Kubernetes-native, multi-framework
   - BentoML: Simple packaging và serving
"""

# ========== PART 1: TRAIN AND SAVE MODEL ==========

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save():
    """Train model và lưu để serving"""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, "iris_model.joblib")
    
    # Save metadata
    metadata = {
        "model_name": "iris_classifier",
        "version": "1.0.0",
        "features": iris.feature_names,
        "classes": iris.target_names.tolist(),
        "accuracy": model.score(X_test, y_test)
    }
    joblib.dump(metadata, "model_metadata.joblib")
    
    print(f"Model saved! Accuracy: {metadata['accuracy']:.4f}")
    return model, metadata

# ========== PART 2: FASTAPI SERVER ==========

"""
Tạo file: serve_model.py với nội dung sau:
"""

FASTAPI_CODE = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np

app = FastAPI(
    title="Iris Classification API",
    description="ML Model serving với FastAPI",
    version="1.0.0"
)

# Load model khi startup
model = joblib.load("iris_model.joblib")
metadata = joblib.load("model_metadata.joblib")

# Request/Response schemas
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="4 features: sepal_length, sepal_width, petal_length, petal_width"
    )
    
    class Config:
        schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }

class PredictResponse(BaseModel):
    prediction: str
    probability: float
    class_probabilities: dict

class BatchPredictRequest(BaseModel):
    instances: List[List[float]]

# Endpoints
@app.get("/")
def root():
    return {"message": "Iris Classification API", "version": metadata["version"]}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        return PredictResponse(
            prediction=metadata["classes"][prediction],
            probability=float(probabilities[prediction]),
            class_probabilities={
                cls: float(prob) 
                for cls, prob in zip(metadata["classes"], probabilities)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    try:
        X = np.array(request.instances)
        predictions = model.predict(X)
        
        return {
            "predictions": [metadata["classes"][p] for p in predictions]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

# ========== PART 3: DOCKERFILE ==========

DOCKERFILE = '''
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY iris_model.joblib .
COPY model_metadata.joblib .
COPY serve_model.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
'''

REQUIREMENTS = '''
fastapi==0.104.1
uvicorn==0.24.0
joblib==1.3.2
scikit-learn==1.3.2
numpy==1.26.2
'''

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Thêm endpoint POST /predict/explain trả về:
       - Prediction
       - Feature importance cho prediction đó
       - Top 2 features ảnh hưởng nhất

BÀI 2: Thêm input validation:
       - Mỗi feature phải trong khoảng hợp lý (ví dụ: sepal_length từ 4-8)
       - Trả về lỗi 400 nếu không hợp lệ

BÀI 3: Build Docker image và chạy:
       - docker build -t iris-api .
       - docker run -p 8000:8000 iris-api
       - Test với curl hoặc Postman

BÀI 4: Thêm caching với Redis:
       - Cache prediction results
       - Invalidate cache khi model update
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== Model Serving Setup ===")
    
    # Step 1: Train and save
    train_and_save()
    
    # Step 2: Create serve_model.py
    with open("serve_model.py", "w") as f:
        f.write(FASTAPI_CODE)
    print("Created: serve_model.py")
    
    # Step 3: Create Dockerfile
    with open("Dockerfile", "w") as f:
        f.write(DOCKERFILE)
    print("Created: Dockerfile")
    
    # Step 4: Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS)
    print("Created: requirements.txt")
    
    print("\n=== Hướng dẫn chạy ===")
    print("1. uvicorn serve_model:app --reload")
    print("2. Mở http://localhost:8000/docs để test API")
