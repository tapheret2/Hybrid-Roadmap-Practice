"""
================================================================
DS JUNIOR - ADVANCED ML: GRADIENT BOOSTING & TIME SERIES
================================================================

Cài đặt: pip install xgboost lightgbm catboost prophet optuna
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Gradient Boosting:
   - XGBoost: Fast, regularization, parallel
   - LightGBM: Faster, leaf-wise growth
   - CatBoost: Native categorical handling

2. Ensemble Methods:
   - Bagging: Random Forest
   - Boosting: XGBoost, LightGBM
   - Stacking: Meta-learner

3. Time Series:
   - Components: Trend, Seasonality, Noise
   - ARIMA: Traditional, univariate
   - Prophet: Facebook, handles holidays
   - Neural Networks: LSTM, Transformer

4. Hyperparameter Tuning:
   - GridSearchCV: Exhaustive search
   - RandomizedSearchCV: Random sampling
   - Optuna: Bayesian optimization
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== XGBOOST ==========

def xgboost_demo():
    """XGBoost with hyperparameter tuning"""
    import xgboost as xgb
    from sklearn.datasets import make_regression
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create DMatrix (XGBoost special format)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,  # learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'lambda': 1,  # L2 regularization
        'alpha': 0,   # L1 regularization
    }
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predict
    preds = model.predict(dtest)
    print(f"XGBoost R2: {r2_score(y_test, preds):.4f}")
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    print(f"Top features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return model

# ========== LIGHTGBM ==========

def lightgbm_demo():
    """LightGBM for classification"""
    import lightgbm as lgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50)]
    )
    
    preds = model.predict(X_test)
    print(f"LightGBM AUC: {roc_auc_score(y_test, preds):.4f}")
    
    return model

# ========== OPTUNA HYPERPARAMETER TUNING ==========

def optuna_tuning():
    """Hyperparameter tuning with Optuna"""
    import optuna
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params

# ========== TIME SERIES WITH PROPHET ==========

def prophet_demo():
    """Time series forecasting with Prophet"""
    from prophet import Prophet
    
    # Generate sample time series
    dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
    np.random.seed(42)
    values = (
        100 +  # baseline
        np.arange(len(dates)) * 0.1 +  # trend
        20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) +  # yearly seasonality
        5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7) +  # weekly seasonality
        np.random.normal(0, 5, len(dates))  # noise
    )
    
    df = pd.DataFrame({'ds': dates, 'y': values})
    
    # Split
    train = df[:-30]
    test = df[-30:]
    
    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train)
    
    # Forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Evaluate
    preds = forecast.tail(30)['yhat'].values
    rmse = np.sqrt(mean_squared_error(test['y'], preds))
    print(f"Prophet RMSE: {rmse:.4f}")
    
    return model, forecast

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Compare XGBoost vs LightGBM vs CatBoost:
       - Same dataset, same params where possible
       - Measure: accuracy, training time, inference time

BÀI 2: Implement custom evaluation metric:
       - Weighted RMSE (penalize recent errors more)
       - Use in XGBoost training

BÀI 3: Build stacking ensemble:
       - Base: RF, XGB, LGB
       - Meta: Logistic Regression
       - Compare to individual models

BÀI 4: Time series cross-validation:
       - Implement TimeSeriesSplit
       - Walk-forward validation
       - Prevent data leakage
"""

if __name__ == "__main__":
    print("=== Advanced ML Demo ===\n")
    # xgboost_demo()
    # lightgbm_demo()
    # optuna_tuning()
    # prophet_demo()
    print("Uncomment functions to run demos!")
