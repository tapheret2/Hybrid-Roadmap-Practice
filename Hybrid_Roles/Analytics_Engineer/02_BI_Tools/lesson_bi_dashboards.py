"""
================================================================
ANALYTICS ENGINEER - MODULE 2: BI TOOLS & DASHBOARDS
================================================================

Analytics Engineers tạo dashboards và reports cho business
Focus: Storytelling with data, actionable insights

Tools: Tableau, Power BI, Looker, Metabase
"""

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. BI Tools Landscape:
   - Tableau: Most powerful, enterprise
   - Power BI: Microsoft ecosystem, cheaper
   - Looker: LookML modeling, Google
   - Metabase: Open-source, simple

2. Dashboard Design Principles:
   - Top-down: Summary → Details
   - F-pattern reading: Important metrics top-left
   - 5-second rule: Key insight visible immediately
   - Progressive disclosure: Don't overwhelm

3. Chart Selection:
   - Comparison: Bar chart
   - Trend: Line chart
   - Composition: Pie/Stacked bar
   - Distribution: Histogram/Box plot
   - Relationship: Scatter plot

4. Metrics Definition:
   - Leading indicators: Predict future
   - Lagging indicators: Measure past
   - North Star Metric: One key metric

5. Stakeholder Communication:
   - Know your audience
   - Tell a story
   - Actionable recommendations
   - Executive summary first
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== METRICS DEFINITIONS ==========

class MetricsCalculator:
    """Calculate business metrics from data"""
    
    def __init__(self, orders_df: pd.DataFrame, customers_df: pd.DataFrame):
        self.orders = orders_df
        self.customers = customers_df
    
    # === REVENUE METRICS ===
    
    def total_revenue(self, start_date=None, end_date=None):
        """Total revenue in period"""
        df = self._filter_dates(self.orders, 'order_date', start_date, end_date)
        return df['amount'].sum()
    
    def average_order_value(self, start_date=None, end_date=None):
        """AOV = Total Revenue / Number of Orders"""
        df = self._filter_dates(self.orders, 'order_date', start_date, end_date)
        return df['amount'].mean()
    
    def revenue_by_segment(self, start_date=None, end_date=None):
        """Revenue breakdown by customer segment"""
        df = self.orders.merge(self.customers, on='customer_id')
        df = self._filter_dates(df, 'order_date', start_date, end_date)
        return df.groupby('segment')['amount'].sum().to_dict()
    
    # === CUSTOMER METRICS ===
    
    def customer_lifetime_value(self, customer_id):
        """CLV for a specific customer"""
        customer_orders = self.orders[self.orders['customer_id'] == customer_id]
        return customer_orders['amount'].sum()
    
    def customer_acquisition_cost(self, marketing_spend, new_customers):
        """CAC = Marketing Spend / New Customers"""
        return marketing_spend / new_customers if new_customers > 0 else 0
    
    def monthly_recurring_revenue(self, month):
        """MRR for subscription businesses"""
        # Simplified: sum of subscription amounts in month
        return self.orders[
            self.orders['order_date'].dt.to_period('M') == month
        ]['amount'].sum()
    
    # === GROWTH METRICS ===
    
    def month_over_month_growth(self):
        """MoM revenue growth"""
        monthly = self.orders.groupby(
            self.orders['order_date'].dt.to_period('M')
        )['amount'].sum()
        return monthly.pct_change().iloc[-1] * 100
    
    def retention_rate(self, cohort_month, months_later):
        """Retention: % customers still active after N months"""
        cohort = self.customers[
            self.customers['signup_date'].dt.to_period('M') == cohort_month
        ]['customer_id']
        
        target_month = pd.Period(cohort_month) + months_later
        active = self.orders[
            self.orders['order_date'].dt.to_period('M') == target_month
        ]['customer_id'].unique()
        
        retained = len(set(cohort) & set(active))
        return (retained / len(cohort)) * 100 if len(cohort) > 0 else 0
    
    # === HELPER ===
    
    def _filter_dates(self, df, date_col, start_date, end_date):
        if start_date:
            df = df[df[date_col] >= start_date]
        if end_date:
            df = df[df[date_col] <= end_date]
        return df

# ========== DASHBOARD DATA PREP ==========

def prepare_executive_dashboard(orders, customers):
    """Prepare data for executive dashboard"""
    
    # Current period
    today = datetime.now()
    current_month_start = today.replace(day=1)
    last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    
    metrics = MetricsCalculator(orders, customers)
    
    return {
        'kpis': {
            'total_revenue_mtd': metrics.total_revenue(start_date=current_month_start),
            'revenue_last_month': metrics.total_revenue(
                start_date=last_month_start,
                end_date=current_month_start - timedelta(days=1)
            ),
            'aov': metrics.average_order_value(start_date=current_month_start),
            'total_orders': len(orders[orders['order_date'] >= current_month_start]),
            'mom_growth': metrics.month_over_month_growth(),
        },
        'revenue_by_segment': metrics.revenue_by_segment(),
        'daily_trend': orders.groupby(
            orders['order_date'].dt.date
        )['amount'].sum().tail(30).to_dict(),
    }

# ========== REPORT TEMPLATE ==========

def generate_weekly_report(data: dict) -> str:
    """Generate weekly report in markdown"""
    
    report = f"""
# Weekly Business Report
**Week of {datetime.now().strftime('%Y-%m-%d')}**

---

## Executive Summary

| Metric | This Week | vs Last Week |
|--------|-----------|--------------|
| Revenue | ${data['kpis']['total_revenue_mtd']:,.2f} | {data['kpis']['mom_growth']:+.1f}% |
| Orders | {data['kpis']['total_orders']} | - |
| AOV | ${data['kpis']['aov']:,.2f} | - |

---

## Key Insights

1. **Revenue Performance**: {
    'Strong' if data['kpis']['mom_growth'] > 10 
    else 'Stable' if data['kpis']['mom_growth'] > -5 
    else 'Declining'
} growth this period.

2. **Top Segment**: {max(data['revenue_by_segment'], key=data['revenue_by_segment'].get)}

---

## Recommendations

- Focus marketing on high-performing segments
- Investigate declining areas
- Monitor customer acquisition costs

---

*Report generated automatically*
"""
    return report

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Tạo Cohort Analysis:
       - Group customers by signup month
       - Track retention over 12 months
       - Visualize as heatmap

BÀI 2: Build Product Analytics Dashboard:
       - Page views, sessions, bounce rate
       - Conversion funnel
       - User segmentation

BÀI 3: Create automated alerting:
       - Revenue drops > 20%
       - Unusual traffic patterns
       - Send Slack notification

BÀI 4: Setup Metabase:
       - Connect to your database
       - Create 3 saved questions
       - Build a dashboard
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== BI & Dashboard Demo ===\n")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    orders = pd.DataFrame({
        'order_id': range(1, 301),
        'customer_id': np.random.randint(1, 51, 300),
        'order_date': np.random.choice(dates, 300),
        'amount': np.random.uniform(50, 500, 300)
    })
    
    customers = pd.DataFrame({
        'customer_id': range(1, 51),
        'segment': np.random.choice(['Enterprise', 'SMB', 'Consumer'], 50),
        'signup_date': pd.to_datetime('2024-01-01') - pd.to_timedelta(np.random.randint(0, 365, 50), unit='D')
    })
    
    # Calculate metrics
    metrics = MetricsCalculator(orders, customers)
    print(f"Total Revenue: ${metrics.total_revenue():,.2f}")
    print(f"AOV: ${metrics.average_order_value():,.2f}")
    print(f"Revenue by Segment: {metrics.revenue_by_segment()}")
    
    # Generate report
    dashboard_data = prepare_executive_dashboard(orders, customers)
    print("\n" + generate_weekly_report(dashboard_data))
