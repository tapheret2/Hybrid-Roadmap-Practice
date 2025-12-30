"""
================================================================
DS INTERN - BÀI 7: MACHINE LEARNING BASICS (SCIKIT-LEARN)
================================================================

Cài đặt: pip install scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Supervised Learning:
   - Regression: Dự đoán giá trị liên tục (giá nhà, lương, ...)
   - Classification: Dự đoán nhãn (spam/not spam, bệnh/không bệnh, ...)

2. Unsupervised Learning:
   - Clustering: Gom nhóm dữ liệu (customer segmentation, ...)

3. ML Workflow:
   Data → Preprocessing → Train/Test Split → Train Model → Evaluate → Deploy

4. Metrics:
   - Regression: MSE, MAE, RMSE, R²
   - Classification: Accuracy, Precision, Recall, F1-score

5. Overfitting vs Underfitting:
   - Overfitting: Model quá phức tạp, học thuộc training data
   - Underfitting: Model quá đơn giản, không học được pattern
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== REGRESSION ==========
print("=" * 50)
print("REGRESSION EXAMPLE: Dự đoán giá nhà")
print("=" * 50)

# Tạo dữ liệu giả lập
np.random.seed(42)
X_reg = np.random.rand(100, 2) * 100  # 2 features: size, bedrooms
y_reg = 50 + 0.3 * X_reg[:, 0] + 20 * X_reg[:, 1] + np.random.randn(100) * 10

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

# Predict
y_pred = reg_model.predict(X_test_scaled)

# Evaluate
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"Coefficients: {reg_model.coef_}")

# ========== CLASSIFICATION ==========
print("\n" + "=" * 50)
print("CLASSIFICATION EXAMPLE: Phân loại Iris")
print("=" * 50)

from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X_clf, y_clf = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")

# Detailed metrics for best model
best_model = models['Random Forest']
y_pred = best_model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Feature Importance (Random Forest)
print("\n--- Feature Importance ---")
for name, importance in zip(iris.feature_names, best_model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# ========== CLUSTERING ==========
print("\n" + "=" * 50)
print("CLUSTERING EXAMPLE: Customer Segmentation")
print("=" * 50)

# Tạo dữ liệu customer
np.random.seed(42)
customers = pd.DataFrame({
    'annual_income': np.concatenate([
        np.random.normal(30, 5, 50),
        np.random.normal(60, 10, 50),
        np.random.normal(100, 15, 50)
    ]),
    'spending_score': np.concatenate([
        np.random.normal(20, 5, 50),
        np.random.normal(50, 10, 50),
        np.random.normal(80, 10, 50)
    ])
})

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customers['cluster'] = kmeans.fit_predict(X_scaled)

print(customers.groupby('cluster').mean())

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Tạo dataset với 3 features và 1 target (regression)
       - Train Linear Regression
       - Tính R² và giải thích ý nghĩa

BÀI 2: Tải dataset Titanic (từ seaborn hoặc kaggle)
       - Preprocessing: Handle missing values, encode categorical
       - Train model để dự đoán survived
       - Tính Precision, Recall, F1

BÀI 3: Thử thay đổi hyperparameters của RandomForest
       - n_estimators: 50, 100, 200
       - max_depth: 3, 5, 10, None
       - So sánh kết quả

BÀI 4 (Nâng cao): Implement Cross-Validation
       - Dùng cross_val_score để đánh giá model robust hơn
"""

# --- TEST ---
if __name__ == "__main__":
    print("\n=== Hoàn thành các bài tập Machine Learning ===")
