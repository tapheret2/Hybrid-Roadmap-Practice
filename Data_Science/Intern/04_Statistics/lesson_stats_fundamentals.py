"""
================================================================
DS INTERN - BÀI 5: STATISTICS FUNDAMENTALS
================================================================

Thống kê là nền tảng của Data Science
"""

import numpy as np
import pandas as pd
from scipy import stats

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Descriptive Statistics: Mô tả dữ liệu (mean, median, std, ...)
2. Probability: Xác suất xảy ra sự kiện
3. Distributions: Normal, Binomial, Poisson, ...
4. Hypothesis Testing: Kiểm định giả thuyết
5. Correlation: Mối tương quan giữa các biến
6. QUAN TRỌNG: Correlation ≠ Causation (tương quan ≠ nhân quả)
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# Sample Data
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)  # IQ distribution

# DESCRIPTIVE STATISTICS
print("=== DESCRIPTIVE STATISTICS ===")
print(f"Mean (Trung bình): {np.mean(data):.2f}")
print(f"Median (Trung vị): {np.median(data):.2f}")
print(f"Mode (Yếu vị): {stats.mode(data.round()).mode}")
print(f"Std (Độ lệch chuẩn): {np.std(data):.2f}")
print(f"Variance (Phương sai): {np.var(data):.2f}")
print(f"Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
print(f"Range: {np.ptp(data):.2f}")  # max - min
print(f"Percentile 25%: {np.percentile(data, 25):.2f}")
print(f"Percentile 75%: {np.percentile(data, 75):.2f}")
print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# PROBABILITY DISTRIBUTIONS
print("\n=== DISTRIBUTIONS ===")

# Normal Distribution
mu, sigma = 100, 15
normal_dist = stats.norm(loc=mu, scale=sigma)
print(f"P(X < 115) = {normal_dist.cdf(115):.4f}")  # CDF
print(f"P(X > 130) = {1 - normal_dist.cdf(130):.4f}")
print(f"P(85 < X < 115) = {normal_dist.cdf(115) - normal_dist.cdf(85):.4f}")

# Z-Score: Đo xem một giá trị cách mean bao nhiêu độ lệch chuẩn
value = 130
z_score = (value - mu) / sigma
print(f"Z-score của {value}: {z_score:.2f}")

# Binomial Distribution (n trials, p probability)
n, p = 10, 0.5
binom_dist = stats.binom(n=n, p=p)
print(f"P(X = 5) with n=10, p=0.5: {binom_dist.pmf(5):.4f}")

# HYPOTHESIS TESTING
print("\n=== HYPOTHESIS TESTING ===")

# T-Test: So sánh trung bình 2 nhóm
group_a = np.random.normal(100, 10, 50)
group_b = np.random.normal(105, 10, 50)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ Reject H0: Có sự khác biệt có ý nghĩa thống kê")
else:
    print("→ Fail to reject H0: Không đủ bằng chứng")

# Chi-Square Test: Kiểm tra mối quan hệ giữa 2 biến categorical
observed = np.array([[50, 30], [20, 40]])  # Tần số quan sát
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

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Cho dataset điểm thi của 2 lớp (A và B)
       - Tính mean, std của mỗi lớp
       - Dùng t-test để kiểm tra xem 2 lớp có khác biệt có ý nghĩa không
"""
def compare_classes():
    class_a = [75, 82, 68, 90, 85, 72, 78, 88, 92, 70]
    class_b = [80, 85, 79, 95, 88, 82, 84, 91, 87, 83]
    # Viết code tại đây
    pass

"""
BÀI 2: Tính Z-score của từng phần tử trong dataset
       Tìm các outliers (|z| > 2)
"""
def find_outliers():
    data = [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 11, 10]
    # Viết code tại đây
    pass

"""
BÀI 3: Cho bảng contingency về giới tính và sở thích mua hàng
       Dùng Chi-square test để kiểm tra xem có mối quan hệ không
       
       |         | Sports | Fashion | Tech |
       |---------|--------|---------|------|
       | Male    |   40   |   20    |  30  |
       | Female  |   25   |   45    |  20  |
"""
def test_independence():
    # Viết code tại đây
    pass

# --- TEST ---
if __name__ == "__main__":
    print("=== Hoàn thành các bài tập Statistics ===")
