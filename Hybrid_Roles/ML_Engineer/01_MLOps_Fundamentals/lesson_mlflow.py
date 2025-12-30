"""
================================================================
ML ENGINEER - MODULE 1: MLOPS FUNDAMENTALS
================================================================

MLOps = Machine Learning + DevOps
Mục tiêu: Đưa ML models từ notebook vào production một cách bài bản

Cài đặt: pip install mlflow scikit-learn pandas
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import json
import os

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. MLOps là gì?
   - Kết hợp ML + DevOps để tự động hóa và quản lý ML lifecycle
   - Bao gồm: Data → Train → Deploy → Monitor → Retrain

2. Experiment Tracking:
   - Lưu lại parameters, metrics, artifacts của mỗi lần train
   - So sánh các experiments để chọn model tốt nhất
   - Tools: MLflow, Weights & Biases, Neptune

3. Model Registry:
   - Quản lý versions của models
   - Staging → Production → Archived
   - Rollback khi cần

4. Reproducibility:
   - Phải reproduce được kết quả bất kỳ lúc nào
   - Cần track: code, data, environment, hyperparameters

5. CI/CD for ML:
   - Continuous Integration: Auto test khi code thay đổi
   - Continuous Deployment: Auto deploy model mới
   - Continuous Training: Auto retrain khi data mới
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== MLFLOW EXPERIMENT TRACKING ==========

def train_with_mlflow():
    """Demo MLflow experiment tracking"""
    
    # Set experiment name
    mlflow.set_experiment("iris_classification")
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Define hyperparameters to test
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="rf_experiment_1"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        # Log metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log artifacts (additional files)
        feature_importance = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        }).to_dict()
        
        with open("feature_importance.json", "w") as f:
            json.dump(feature_importance, f)
        mlflow.log_artifact("feature_importance.json")
        os.remove("feature_importance.json")
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")
        
        return model

# ========== EXPERIMENT COMPARISON ==========

def compare_experiments():
    """So sánh nhiều experiments"""
    
    mlflow.set_experiment("iris_classification")
    
    # Test multiple hyperparameters
    param_grid = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"experiment_{i+1}"):
            params["random_state"] = 42
            mlflow.log_params(params)
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Experiment {i+1}: {params} → Accuracy: {accuracy:.4f}")

# ========== MODEL REGISTRY ==========

def register_best_model():
    """Đăng ký model vào registry"""
    
    # Tìm run có accuracy cao nhất
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris_classification")
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            print(f"Best run: {best_run.info.run_id}")
            print(f"Accuracy: {best_run.data.metrics['accuracy']}")
            
            # Register model
            model_uri = f"runs:/{best_run.info.run_id}/model"
            # mlflow.register_model(model_uri, "iris_classifier")

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Tạo một experiment mới với tên "house_price_prediction"
       - Train 3 models khác nhau (Linear Regression, Random Forest, XGBoost)
       - Log parameters, metrics (MSE, MAE, R2), và model
       - So sánh và chọn model tốt nhất

BÀI 2: Thêm logging cho:
       - Training time
       - Data version (hash của dataset)
       - Model size (bytes)

BÀI 3: Tạo custom MLflow plugin để log:
       - Confusion matrix như một artifact image
       - Feature importance plot

BÀI 4 (Nâng cao): Setup MLflow tracking server:
       - Chạy: mlflow server --host 0.0.0.0 --port 5000
       - Connect từ code: mlflow.set_tracking_uri("http://localhost:5000")
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== MLOps Fundamentals with MLflow ===")
    print("Chạy: mlflow ui để xem dashboard tại http://localhost:5000")
    # train_with_mlflow()
    # compare_experiments()
    # register_best_model()
