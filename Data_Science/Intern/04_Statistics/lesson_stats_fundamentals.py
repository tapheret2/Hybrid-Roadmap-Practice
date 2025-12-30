"""
================================================================
DS INTERN - LESSON 5: STATISTICS FUNDAMENTALS
================================================================

Statistics is the foundation of Data Science
"""

import numpy as np
import pandas as pd
from scipy import stats

# --- 1. THEORY ---
"""
1. Descriptive Statistics: Summarize data (mean, median, std, ...)
2. Probability: Likelihood of events occurring
3. Distributions: Normal, Binomial, Poisson, ...
4. Hypothesis Testing: Test assumptions about data
5. Correlation: Relationship between variables
6. IMPORTANT: Correlation ≠ Causation
"""

# --- 2. CODE SAMPLE ---

# Sample Data
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)  # IQ distribution

# DESCRIPTIVE STATISTICS
print("=== DESCRIPTIVE STATISTICS ===")
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Mode: {stats.mode(data.round()).mode}")
print(f"Standard Deviation: {np.std(data):.2f}")
print(f"Variance: {np.var(data):.2f}")
print(f"Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
print(f"Range: {np.ptp(data):.2f}")  # max - min
print(f"25th Percentile: {np.percentile(data, 25):.2f}")
print(f"75th Percentile: {np.percentile(data, 75):.2f}")
print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# PROBABILITY DISTRIBUTIONS
print("\n=== DISTRIBUTIONS ===")

# Normal Distribution
mu, sigma = 100, 15
normal_dist = stats.norm(loc=mu, scale=sigma)
print(f"P(X < 115) = {normal_dist.cdf(115):.4f}")  # CDF
print(f"P(X > 130) = {1 - normal_dist.cdf(130):.4f}")
print(f"P(85 < X < 115) = {normal_dist.cdf(115) - normal_dist.cdf(85):.4f}")

# Z-Score: How many standard deviations from mean
value = 130
z_score = (value - mu) / sigma
print(f"Z-score of {value}: {z_score:.2f}")

# Binomial Distribution (n trials, p probability)
n, p = 10, 0.5
binom_dist = stats.binom(n=n, p=p)
print(f"P(X = 5) with n=10, p=0.5: {binom_dist.pmf(5):.4f}")

# HYPOTHESIS TESTING
print("\n=== HYPOTHESIS TESTING ===")

# T-Test: Compare means of 2 groups
group_a = np.random.normal(100, 10, 50)
group_b = np.random.normal(105, 10, 50)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ Reject H0: Statistically significant difference")
else:
    print("→ Fail to reject H0: Not enough evidence")

# Chi-Square Test: Test relationship between 2 categorical variables
observed = np.array([[50, 30], [20, 40]])  # Observed frequencies
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-square: {chi2:.4f}, P-value: {p_value:.4f}")

# CORRELATION
print("\n=== CORRELATION ===")

x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5  # y = 2x + noise

# Pearson Correlation (linear relationship)
pearson_r, p_val = stats.pearsonr(x, y)
print(f"Pearson correlation: {pearson_r:.4f} (p-value: {p_val:.4f})")

# Spearman Correlation (monotonic relationship)
spearman_r, p_val = stats.spearmanr(x, y)
print(f"Spearman correlation: {spearman_r:.4f}")

# Interpretation
# |r| < 0.3: Weak
# 0.3 <= |r| < 0.7: Moderate
# |r| >= 0.7: Strong

# --- 3. EXERCISES ---
"""
EXERCISE 1: Given test scores from 2 classes (A and B)
           - Calculate mean, std for each class
           - Use t-test to check if there's significant difference
"""
def compare_classes():
    class_a = [75, 82, 68, 90, 85, 72, 78, 88, 92, 70]
    class_b = [80, 85, 79, 95, 88, 82, 84, 91, 87, 83]
    # Write code here
    pass

"""
EXERCISE 2: Calculate Z-score for each element in dataset
           Find outliers (|z| > 2)
"""
def find_outliers():
    data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 11, 10]
    # Write code here
    pass

"""
EXERCISE 3: Given contingency table of gender vs shopping preference
           Use Chi-square test to check if there's relationship
           
           |         | Sports | Fashion | Tech |
           |---------|--------|---------|------|
           | Male    |   40   |   20    |  30  |
           | Female  |   25   |   45    |  20  |
"""
def test_independence():
    # Write code here
    pass

# --- TEST ---
if __name__ == "__main__":
    print("=== Complete the Statistics exercises ===")
