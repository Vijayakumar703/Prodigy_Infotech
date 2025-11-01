# Task 1: Bar Chart and Histogram Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- Generate random dataset ---
np.random.seed(42)
df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female', 'Other'], size=300, p=[0.48, 0.48, 0.04]),
    'Age': np.clip(np.random.normal(loc=30, scale=8, size=300).astype(int), 18, 65)
})

# --- 1) Bar chart for categorical variable (Gender) ---
counts = df['Gender'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(counts.index, counts.values, color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_title('Distribution of Gender', fontsize=14)
ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
fig.tight_layout()
fig.savefig('gender_bar_chart.png', dpi=200)
plt.show()

# --- Download bar chart image ---
files.download('gender_bar_chart.png')

# --- 2) Histogram for continuous variable (Age) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Age'], bins=10, edgecolor='black', color='#8172B2')
ax.set_title('Distribution of Age', fontsize=14)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
fig.tight_layout()
fig.savefig('age_histogram.png', dpi=200)
plt.show()

# --- Download histogram image ---
files.download('age_histogram.png')
