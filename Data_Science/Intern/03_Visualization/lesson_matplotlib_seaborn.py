"""
================================================================
DS INTERN - LESSON 4: DATA VISUALIZATION (MATPLOTLIB & SEABORN)
================================================================

Install: pip install matplotlib seaborn
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- 1. THEORY ---
"""
1. Matplotlib: Basic library, flexible, lots of customization
2. Seaborn: Wrapper around Matplotlib, prettier, easier for statistics
3. Figure & Axes: Figure is the canvas, Axes is the plot area
4. Choosing the right chart:
   - Distribution: Histogram, KDE, Boxplot
   - Relationship: Scatter, Line, Heatmap (correlation)
   - Comparison: Bar, Grouped Bar
   - Composition: Pie, Stacked Bar
"""

# Sample Data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 100),
    'income': np.random.randint(3000, 15000, 100),
    'spending': np.random.randint(1000, 5000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# --- 2. CODE SAMPLE ---

# 1. HISTOGRAM - Distribution
def plot_histogram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matplotlib
    axes[0].hist(df['income'], bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Income Distribution (Matplotlib)')
    axes[0].set_xlabel('Income')
    axes[0].set_ylabel('Frequency')
    
    # Seaborn
    sns.histplot(data=df, x='income', kde=True, ax=axes[1])
    axes[1].set_title('Income Distribution (Seaborn)')
    
    plt.tight_layout()
    plt.savefig('histogram.png', dpi=100)
    # plt.show()

# 2. BOXPLOT - Compare distributions across groups
def plot_boxplot():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='region', y='income', hue='gender', palette='Set2')
    ax.set_title('Income by Region and Gender')
    plt.savefig('boxplot.png', dpi=100)
    # plt.show()

# 3. SCATTER PLOT - Relationship between 2 variables
def plot_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Basic Scatter
    axes[0].scatter(df['age'], df['income'], alpha=0.5, c='steelblue')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Income')
    axes[0].set_title('Age vs Income')
    
    # Seaborn with regression line
    sns.regplot(data=df, x='age', y='income', ax=axes[1], scatter_kws={'alpha': 0.5})
    axes[1].set_title('Age vs Income (with regression)')
    
    plt.tight_layout()
    plt.savefig('scatter.png', dpi=100)
    # plt.show()

# 4. BAR CHART - Comparison
def plot_bar():
    region_income = df.groupby('region')['income'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(region_income.index, region_income.values, color='steelblue')
    ax.bar_label(bars, fmt='%.0f')
    ax.set_xlabel('Average Income')
    ax.set_title('Average Income by Region')
    plt.savefig('bar.png', dpi=100)
    # plt.show()

# 5. HEATMAP - Correlation Matrix
def plot_heatmap():
    corr = df[['age', 'income', 'spending']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')
    plt.savefig('heatmap.png', dpi=100)
    # plt.show()

# 6. MULTIPLE SUBPLOTS - Dashboard
def plot_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(data=df, x='age', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')
    
    sns.boxplot(data=df, x='region', y='income', ax=axes[0, 1])
    axes[0, 1].set_title('Income by Region')
    
    sns.scatterplot(data=df, x='age', y='spending', hue='gender', ax=axes[1, 0])
    axes[1, 0].set_title('Age vs Spending')
    
    region_counts = df['region'].value_counts()
    axes[1, 1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Region Distribution')
    
    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=100)
    # plt.show()

# --- 3. EXERCISES ---
"""
EXERCISE 1: Create grouped bar chart comparing average spending 
           between Male and Female by Region

EXERCISE 2: Create FacetGrid:
           - One subplot per region
           - Scatter plot of age vs income
           Hint: sns.FacetGrid or sns.relplot

EXERCISE 3: Create 4-panel dashboard:
           - Top-left: KDE plot of income
           - Top-right: Violin plot income by gender
           - Bottom-left: Pair plot (age, income, spending)
           - Bottom-right: Count plot of region
"""

# --- TEST ---
if __name__ == "__main__":
    print("=== Run plot functions to see results ===")
    # plot_histogram()
    # plot_boxplot()
    # plot_scatter()
    # plot_bar()
    # plot_heatmap()
    # plot_dashboard()
    print("Uncomment functions and run to see plots!")
