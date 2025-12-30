"""
================================================================
DS JUNIOR - MLOPS: MODEL DEPLOYMENT & PRODUCTION
================================================================

Cài đặt: pip install fastapi mlflow docker
"""

import pickle
import joblib
from datetime import datetime
from typing import List, Dict, Any
import json

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Model Serialization:
   - pickle/joblib: Python objects
   - ONNX: Cross-platform, optimized
   - TorchScript: PyTorch deployment

2. Serving Patterns:
   - Batch: Scheduled predictions
   - Online: Real-time API
   - Streaming: Continuous data

3. MLOps Lifecycle:
   - Development: Experiment, train
   - Deployment: Package, serve
   - Monitoring: Drift, performance
   - Retraining: Continuous improvement

4. Best Practices:
   - Version control models
   - A/B testing
   - Canary deployments
   - Rollback capability
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== MODEL WRAPPER ==========

class ModelWrapper:
    """Production-ready model wrapper"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
        }
    
    def preprocess(self, data: Dict[str, Any]) -> Any:
        """Preprocess input data"""
        if self.preprocessor:
            return self.preprocessor.transform(data)
        return data
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with preprocessing and postprocessing"""
        start_time = datetime.now()
        
        try:
            processed = self.preprocess(data)
            prediction = self.model.predict(processed)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed)
            else:
                probabilities = None
            
            return {
                'prediction': prediction.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'latency_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'model_version': self.metadata['version']
            }
        except Exception as e:
            return {
                'error': str(e),
                'latency_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
    
    def save(self, path: str):
        """Save model with metadata"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metadata': self.metadata
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from path"""
        data = joblib.load(path)
        wrapper = cls(data['model'], data['preprocessor'])
        wrapper.metadata = data['metadata']
        return wrapper

# ========== FASTAPI SERVING ==========

FASTAPI_CODE = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model on startup
model_wrapper = None

@app.on_event("startup")
async def load_model():
    global model_wrapper
    model_wrapper = ModelWrapper.load("model.joblib")

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class BatchRequest(BaseModel):
    instances: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": model_wrapper.metadata["version"]
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    return model_wrapper.predict(request.features)

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = [model_wrapper.predict(instance) for instance in request.instances]
    return {"predictions": results}

@app.get("/metadata")
def get_metadata():
    return model_wrapper.metadata
'''

# ========== CI/CD FOR ML ==========

GITHUB_ACTIONS_ML = '''
name: ML Pipeline

on:
  push:
    paths:
      - 'models/**'
      - 'data/**'
  schedule:
    - cron: '0 0 * * 0'  # Weekly retraining

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run training
        run: python train.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      
      - name: Evaluate model
        run: python evaluate.py
      
      - name: Check metrics threshold
        run: |
          ACCURACY=$(cat metrics.json | jq '.accuracy')
          if (( $(echo "$ACCURACY < 0.85" | bc -l) )); then
            echo "Model accuracy below threshold"
            exit 1
          fi
      
      - name: Upload model
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: model.joblib

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: model
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ml-api --image gcr.io/$PROJECT/ml-api
'''

# ========== A/B TESTING ==========

class ABTestManager:
    """Simple A/B testing for models"""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1"
    
    def select_model(self, user_id: str) -> str:
        """Select model based on user_id hash"""
        import hashlib
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        rand = (hash_value % 100) / 100
        
        cumulative = 0
        for model_name, weight in self.weights.items():
            cumulative += weight
            if rand < cumulative:
                return model_name
        return list(self.models.keys())[-1]
    
    def predict(self, user_id: str, data: Dict) -> Dict:
        model_name = self.select_model(user_id)
        model = self.models[model_name]
        
        result = model.predict(data)
        result['model_variant'] = model_name
        
        return result

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Implement model versioning:
       - Store multiple versions
       - Rollback capability
       - Compare versions

BÀI 2: Add model monitoring:
       - Log predictions
       - Track latency distribution
       - Detect concept drift

BÀI 3: Setup canary deployment:
       - 10% traffic to new model
       - Gradually increase if metrics good
       - Auto rollback if degradation

BÀI 4: Implement shadow mode:
       - New model runs in parallel
       - Compare predictions without affecting users
       - Switch when confident
"""

if __name__ == "__main__":
    print("=== MLOps Demo ===\n")
    
    # Demo model wrapper
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(iris.data, iris.target)
    
    wrapper = ModelWrapper(model)
    wrapper.metadata['features'] = iris.feature_names
    wrapper.metadata['classes'] = iris.target_names.tolist()
    
    # Test prediction
    test_data = {'features': iris.data[0].reshape(1, -1)}
    result = wrapper.predict(test_data)
    print(f"Prediction: {result}")
    
    # Save and load
    wrapper.save("model.joblib")
    loaded = ModelWrapper.load("model.joblib")
    print(f"Loaded model version: {loaded.metadata['version']}")
