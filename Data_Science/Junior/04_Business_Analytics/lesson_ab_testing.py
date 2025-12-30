"""
================================================================
DS JUNIOR - BUSINESS ANALYTICS: A/B TESTING & METRICS
================================================================

Kết nối Data Science với Business Impact
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. A/B Testing:
   - Hypothesis: H0 (no difference) vs H1 (difference)
   - Sample size calculation
   - Statistical significance (p-value < 0.05)
   - Practical significance (effect size)

2. Business Metrics:
   - North Star Metric: One key success metric
   - Leading vs Lagging indicators
   - Proxy metrics

3. Common Metrics:
   - Acquisition: CAC, Traffic sources
   - Activation: Signup rate, Time to first action
   - Retention: D1/D7/D30 retention
   - Revenue: ARPU, LTV, MRR
   - Referral: NPS, Viral coefficient

4. Causal Inference:
   - Correlation ≠ Causation
   - Confounding variables
   - Instrumental variables
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== SAMPLE SIZE CALCULATION ==========

def calculate_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:
    """
    Calculate required sample size per variant
    
    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
        min_detectable_effect: Minimum relative change to detect (e.g., 0.10 for 10% lift)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    p_avg = (p1 + p2) / 2
    
    n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (p2 - p1)**2
    
    return int(np.ceil(n))

# ========== A/B TEST ANALYSIS ==========

@dataclass
class ABTestResult:
    control_rate: float
    treatment_rate: float
    relative_lift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int

def analyze_ab_test(
    control_conversions: int,
    control_visitors: int,
    treatment_conversions: int,
    treatment_visitors: int,
    alpha: float = 0.05
) -> ABTestResult:
    """
    Analyze A/B test results using chi-square test
    """
    # Conversion rates
    control_rate = control_conversions / control_visitors
    treatment_rate = treatment_conversions / treatment_visitors
    
    # Relative lift
    relative_lift = (treatment_rate - control_rate) / control_rate
    
    # Chi-square test
    observed = np.array([
        [control_conversions, control_visitors - control_conversions],
        [treatment_conversions, treatment_visitors - treatment_conversions]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Confidence interval for difference
    se = np.sqrt(
        control_rate * (1 - control_rate) / control_visitors +
        treatment_rate * (1 - treatment_rate) / treatment_visitors
    )
    z = stats.norm.ppf(1 - alpha/2)
    diff = treatment_rate - control_rate
    ci = (diff - z * se, diff + z * se)
    
    return ABTestResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        relative_lift=relative_lift,
        p_value=p_value,
        confidence_interval=ci,
        is_significant=p_value < alpha,
        sample_size_control=control_visitors,
        sample_size_treatment=treatment_visitors
    )

# ========== RETENTION ANALYSIS ==========

def calculate_retention(
    events: pd.DataFrame,
    user_col: str = 'user_id',
    date_col: str = 'event_date',
    signup_date_col: str = 'signup_date'
) -> pd.DataFrame:
    """
    Calculate retention cohort analysis
    """
    events = events.copy()
    events[date_col] = pd.to_datetime(events[date_col])
    events[signup_date_col] = pd.to_datetime(events[signup_date_col])
    
    # Day since signup
    events['day_number'] = (events[date_col] - events[signup_date_col]).dt.days
    
    # Cohort (signup month)
    events['cohort'] = events[signup_date_col].dt.to_period('M')
    
    # Count users per cohort per day
    retention = events.groupby(['cohort', 'day_number'])[user_col].nunique().unstack(fill_value=0)
    
    # Calculate percentage
    cohort_size = retention[0]
    retention_pct = retention.div(cohort_size, axis=0) * 100
    
    return retention_pct

# ========== LTV CALCULATION ==========

def calculate_ltv(
    orders: pd.DataFrame,
    user_col: str = 'user_id',
    revenue_col: str = 'revenue',
    date_col: str = 'order_date',
    prediction_months: int = 12
) -> pd.DataFrame:
    """
    Calculate Customer Lifetime Value
    """
    orders = orders.copy()
    orders[date_col] = pd.to_datetime(orders[date_col])
    
    # Per-user metrics
    user_metrics = orders.groupby(user_col).agg({
        revenue_col: ['sum', 'mean', 'count'],
        date_col: ['min', 'max']
    })
    user_metrics.columns = ['total_revenue', 'avg_order', 'order_count', 'first_order', 'last_order']
    
    # Calculate lifespan
    user_metrics['lifespan_days'] = (user_metrics['last_order'] - user_metrics['first_order']).dt.days
    user_metrics['lifespan_months'] = user_metrics['lifespan_days'] / 30
    
    # Monthly revenue
    user_metrics['monthly_revenue'] = np.where(
        user_metrics['lifespan_months'] > 0,
        user_metrics['total_revenue'] / user_metrics['lifespan_months'],
        user_metrics['total_revenue']
    )
    
    # Predicted LTV
    user_metrics['predicted_ltv'] = user_metrics['monthly_revenue'] * prediction_months
    
    return user_metrics

# ========== BUSINESS REPORT ==========

def generate_weekly_metrics(
    events: pd.DataFrame,
    orders: pd.DataFrame,
    week_start: datetime
) -> dict:
    """
    Generate weekly business metrics report
    """
    week_end = week_start + timedelta(days=7)
    
    # Filter to week
    weekly_events = events[
        (events['event_date'] >= week_start) & 
        (events['event_date'] < week_end)
    ]
    weekly_orders = orders[
        (orders['order_date'] >= week_start) & 
        (orders['order_date'] < week_end)
    ]
    
    # Calculate metrics
    metrics = {
        'period': f"{week_start.date()} to {week_end.date()}",
        'unique_users': weekly_events['user_id'].nunique(),
        'total_sessions': len(weekly_events[weekly_events['event'] == 'session']),
        'signups': len(weekly_events[weekly_events['event'] == 'signup']),
        'orders': len(weekly_orders),
        'revenue': weekly_orders['revenue'].sum(),
        'aov': weekly_orders['revenue'].mean(),
        'conversion_rate': len(weekly_orders) / weekly_events['user_id'].nunique() * 100
            if weekly_events['user_id'].nunique() > 0 else 0
    }
    
    return metrics

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Implement sequential A/B testing:
       - Early stopping if clear winner
       - Handle peeking problem

BÀI 2: Build cohort analysis dashboard:
       - Monthly cohorts
       - D1, D7, D30 retention
       - Visualize as heatmap

BÀI 3: Calculate marketing ROI:
       - CAC per channel
       - LTV by channel
       - Payback period

BÀI 4: Implement causal analysis:
       - Difference-in-differences
       - Synthetic control
       - Measure true impact
"""

if __name__ == "__main__":
    print("=== Business Analytics Demo ===\n")
    
    # Sample size calculation
    n = calculate_sample_size(
        baseline_rate=0.10,
        min_detectable_effect=0.10
    )
    print(f"Required sample size per variant: {n:,}")
    
    # A/B Test analysis
    result = analyze_ab_test(
        control_conversions=1200,
        control_visitors=10000,
        treatment_conversions=1380,
        treatment_visitors=10000
    )
    print(f"\nA/B Test Results:")
    print(f"  Control rate: {result.control_rate:.2%}")
    print(f"  Treatment rate: {result.treatment_rate:.2%}")
    print(f"  Relative lift: {result.relative_lift:.2%}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Significant: {result.is_significant}")
    print(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
