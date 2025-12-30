"""
================================================================
ML ENGINEER - MODULE 3: MONITORING & FEATURE STORES
================================================================

Mục tiêu: Giám sát model trong production và quản lý features

Cài đặt: pip install evidently feast pandas scikit-learn
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Model Monitoring:
   - Data Drift: Input data thay đổi so với training data
   - Concept Drift: Mối quan hệ X→Y thay đổi
   - Performance Degradation: Accuracy giảm theo thời gian

2. Monitoring Metrics:
   - Statistical: PSI, KL divergence, chi-square
   - Performance: Accuracy, latency, throughput
   - Business: Conversion rate, revenue impact

3. Feature Stores:
   - Centralized repository cho features
   - Consistent features giữa training và serving
   - Tools: Feast, Tecton, Databricks Feature Store

4. A/B Testing:
   - So sánh model mới vs model cũ
   - Statistical significance
   - Gradual rollout (canary deployment)

5. Alerting:
   - Threshold-based alerts
   - Anomaly detection
   - PagerDuty, Slack integration
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== DATA DRIFT DETECTION ==========

def detect_drift_manual(reference_data, current_data, threshold=0.1):
    """
    Phát hiện data drift bằng PSI (Population Stability Index)
    PSI < 0.1: No drift
    0.1 <= PSI < 0.25: Moderate drift
    PSI >= 0.25: Significant drift
    """
    def calculate_psi(expected, actual, bins=10):
        """Calculate PSI for a single feature"""
        # Create bins from reference data
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Calculate proportions
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        expected_pct = expected_counts / len(expected) + 0.0001
        actual_pct = actual_counts / len(actual) + 0.0001
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi
    
    results = {}
    for col in reference_data.columns:
        if reference_data[col].dtype in ['int64', 'float64']:
            psi = calculate_psi(reference_data[col].values, current_data[col].values)
            results[col] = {
                'psi': psi,
                'drift_detected': psi >= threshold,
                'severity': 'none' if psi < 0.1 else 'moderate' if psi < 0.25 else 'significant'
            }
    
    return results

# Demo drift detection
def demo_drift_detection():
    """Demo phát hiện drift"""
    np.random.seed(42)
    
    # Reference data (training)
    reference = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 10, 1000)
    })
    
    # Current data with drift
    current = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 1000),  # Shifted mean, larger std
        'feature_2': np.random.normal(5, 2, 1000),      # No drift
        'feature_3': np.random.uniform(2, 12, 1000)     # Shifted range
    })
    
    results = detect_drift_manual(reference, current)
    print("=== Drift Detection Results ===")
    for feature, result in results.items():
        print(f"{feature}: PSI={result['psi']:.4f}, Severity={result['severity']}")

# ========== FEATURE STORE CONCEPT ==========

class SimpleFeatureStore:
    """
    Simple in-memory Feature Store for learning purposes
    Production: Use Feast, Tecton, or similar
    """
    
    def __init__(self):
        self.features = {}
        self.metadata = {}
    
    def register_feature(self, name, description, dtype, source):
        """Register a feature definition"""
        self.metadata[name] = {
            'description': description,
            'dtype': dtype,
            'source': source,
            'created_at': datetime.now().isoformat()
        }
    
    def ingest(self, entity_id, features: dict, timestamp=None):
        """Ingest feature values for an entity"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if entity_id not in self.features:
            self.features[entity_id] = []
        
        self.features[entity_id].append({
            'timestamp': timestamp,
            'features': features
        })
    
    def get_features(self, entity_id, feature_names=None, as_of=None):
        """Get features for an entity (point-in-time lookup)"""
        if entity_id not in self.features:
            return None
        
        records = self.features[entity_id]
        
        if as_of:
            records = [r for r in records if r['timestamp'] <= as_of]
        
        if not records:
            return None
        
        # Get latest
        latest = max(records, key=lambda x: x['timestamp'])
        
        if feature_names:
            return {k: v for k, v in latest['features'].items() if k in feature_names}
        return latest['features']
    
    def get_training_data(self, entity_ids, feature_names, timestamps):
        """Get point-in-time correct training data"""
        data = []
        for entity_id, ts in zip(entity_ids, timestamps):
            features = self.get_features(entity_id, feature_names, as_of=ts)
            if features:
                features['entity_id'] = entity_id
                features['timestamp'] = ts
                data.append(features)
        return pd.DataFrame(data)

def demo_feature_store():
    """Demo Feature Store usage"""
    fs = SimpleFeatureStore()
    
    # Register features
    fs.register_feature('user_total_purchases', 'Total purchase amount', 'float', 'transactions_db')
    fs.register_feature('user_avg_session_time', 'Average session duration', 'float', 'analytics_db')
    fs.register_feature('user_days_since_last_order', 'Days since last order', 'int', 'orders_db')
    
    # Ingest historical data
    base_time = datetime.now() - timedelta(days=30)
    for day in range(30):
        ts = base_time + timedelta(days=day)
        fs.ingest('user_001', {
            'user_total_purchases': 100 + day * 10,
            'user_avg_session_time': 5.0 + np.random.randn(),
            'user_days_since_last_order': 30 - day
        }, timestamp=ts)
    
    # Get latest features for serving
    print("=== Feature Store Demo ===")
    latest = fs.get_features('user_001')
    print(f"Latest features: {latest}")
    
    # Get point-in-time features for training
    historical = fs.get_features('user_001', as_of=base_time + timedelta(days=15))
    print(f"Historical (day 15): {historical}")

# ========== A/B TESTING ==========

def ab_test_analysis(control_conversions, control_total, treatment_conversions, treatment_total):
    """
    Phân tích kết quả A/B test
    """
    from scipy import stats
    
    # Conversion rates
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total
    
    # Z-test for proportions
    pooled_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
    z_score = (treatment_rate - control_rate) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Lift
    lift = (treatment_rate - control_rate) / control_rate * 100
    
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'lift': lift,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def demo_ab_test():
    """Demo A/B testing"""
    # Model A (control): 1000 predictions, 120 conversions
    # Model B (treatment): 1000 predictions, 145 conversions
    
    result = ab_test_analysis(120, 1000, 145, 1000)
    
    print("=== A/B Test Results ===")
    print(f"Control conversion: {result['control_rate']:.2%}")
    print(f"Treatment conversion: {result['treatment_rate']:.2%}")
    print(f"Lift: {result['lift']:.2f}%")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Statistically significant: {result['significant']}")

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Implement concept drift detection:
       - So sánh performance metrics (accuracy) giữa các time windows
       - Alert khi accuracy giảm > 5%

BÀI 2: Extend SimpleFeatureStore:
       - Thêm method để list tất cả entities
       - Thêm time-based aggregations (last_7days_avg, last_30days_sum)
       - Thêm caching với TTL

BÀI 3: Setup Feast (production feature store):
       - Install: pip install feast
       - Tạo feature repository
       - Define features trong Python
       - Materialize features

BÀI 4: Implement monitoring dashboard:
       - Log predictions và actual outcomes
       - Tính rolling accuracy periodically
       - Visualize drift metrics over time
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== ML Monitoring & Feature Stores ===\n")
    demo_drift_detection()
    print()
    demo_feature_store()
    print()
    demo_ab_test()
