# Task-02: Exploratory Data Analysis (EDA) - Titanic Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files

# --- Step 1: Load Dataset ---
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# --- Step 2: Basic Information ---
print("Shape of dataset:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# --- Step 3: Handle Missing Values ---
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# --- Step 4: Remove Duplicates ---
df.drop_duplicates(inplace=True)

# --- Step 5: Summary Statistics ---
print("\nStatistical Summary:\n", df.describe())

# --- Step 6: Univariate Analysis ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='Set2')
plt.title('Survival Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=20, kde=True, color='orange')
plt.title('Age Distribution')
plt.show()

# --- Step 7: Bivariate Analysis ---
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Gender')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df, palette='coolwarm')
plt.title('Survival Rate by Passenger Class')
plt.show()

# --- Step 8: Correlation Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap')
plt.show()

# --- Step 9: Save and Download Cleaned Dataset ---
df.to_csv('cleaned_titanic.csv', index=False)
print("✅ Cleaned dataset saved as cleaned_titanic.csv")
files.download('cleaned_titanic.csv')
