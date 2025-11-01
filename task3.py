# Task 03 - Decision Tree Classifier
# Prodigy InfoTech Data Science Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import files

# --- Step 1: Load Dataset ---
data = pd.read_csv('bank.csv', sep=';')   # Upload "bank.csv" before running this cell

# --- Step 2: Explore Data ---
print("First 5 rows:\n", data.head())
print("\nColumns:\n", list(data.columns))
print("\nMissing Values:\n", data.isnull().sum())

# --- Step 3: Encode Categorical Variables ---
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])

# --- Step 4: Split Features and Target ---
X = data.drop('y', axis=1)
y = data['y']

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 6: Train Decision Tree Classifier ---
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --- Step 7: Predictions ---
y_pred = model.pre_
