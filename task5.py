# -----------------------------------------------------------
# Task-05: Traffic Accident Data Analysis (FARS 2016 Format)
# Internship: Prodigy InfoTech
# Dataset: acc_16.csv.zip
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, os
from google.colab import files

# --- Step 1: Extract & Load Dataset ---
zip_file = "acc_16.csv.zip"
if not os.path.exists("acc_16.csv"):
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall()
        print("✅ Zip file extracted successfully!")

df = pd.read_csv("acc_16.csv", low_memory=False)
print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("First 10 columns:", list(df.columns[:10]))

# --- Step 2: Select & Clean Key Columns ---
columns = ['URBANICITY', 'WEATHR_IM', 'LGTCON_IM', 'MAXSEV_IM', 'VE_TOTAL', 'REGION']
df = df[columns].dropna()
print("\n✅ Selected columns for analysis:", columns)

# --- Step 3: Map Encoded Columns ---
df['URBANICITY'] = df['URBANICITY'].map({1: 'Rural', 2: 'Urban'})
df['WEATHR_IM'] = df['WEATHR_IM'].map({
    1:'Clear', 2:'Rain', 3:'Sleet/Hail', 4:'Snow', 5:'Fog/Smog',
    6:'Severe Wind', 7:'Blowing Sand/Dust', 8:'Other', 9:'Unknown'
}).fillna('Unknown')
df['LGTCON_IM'] = df['LGTCON_IM'].map({
    1:'Daylight', 2:'Dark-Lighted', 3:'Dark-Unlighted',
    4:'Dawn', 5:'Dusk', 6:'Other', 7:'Unknown'
}).fillna('Unknown')
df['REGION'] = df['REGION'].map({1:'Northeast', 2:'Midwest', 3:'South', 4:'West'}).fillna('Unknown')

# --- Step 4: Visualization ---
sns.set(style="whitegrid")

# 1️⃣ Urban vs Rural
plt.figure(figsize=(6,4))
sns.countplot(x='URBANICITY', data=df, palette='Set2')
plt.title('🚗 Accidents by Area Type')
plt.tight_layout(); plt.savefig('urbanicity.png', dpi=200); plt.show()

# 2️⃣ Weather Condition
plt.figure(figsize=(8,5))
sns.countplot(y='WEATHR_IM', data=df, order=df['WEATHR_IM'].value_counts().index, palette='coolwarm')
plt.title('🌦️ Accidents by Weather Condition')
plt.tight_layout(); plt.savefig('weather.png', dpi=200); plt.show()

# 3️⃣ Light Condition
plt.figure(figsize=(8,5))
sns.countplot(y='LGTCON_IM', data=df, order=df['LGTCON_IM'].value_counts().index, palette='magma')
plt.title('💡 Accidents by Light Condition')
plt.tight_layout(); plt.savefig('light.png', dpi=200); plt.show()

# 4️⃣ Severity
plt.figure(figsize=(6,4))
sns.countplot(x='MAXSEV_IM', data=df, palette='plasma')
plt.title('⚠️ Distribution of Accident Severity')
plt.tight_layout(); plt.savefig('severity.png', dpi=200); plt.show()

# 5️⃣ Vehicles Involved
plt.figure(figsize=(6,4))
sns.histplot(df['VE_TOTAL'], bins=15, kde=True, color='skyblue')
plt.title('🚘 Distribution of Vehicles per Accident')
plt.tight_layout(); plt.savefig('vehicles.png', dpi=200); plt.show()

# 6️⃣ U.S. Region
plt.figure(figsize=(7,4))
sns.countplot(x='REGION', data=df, palette='viridis')
plt.title('🗺️ Accidents by U.S. Region')
plt.tight_layout(); plt.savefig('region.png', dpi=200); plt.show()

# --- Step 5: Save Cleaned Data ---
df.to_csv('fars2016_cleaned.csv', index=False)
print("✅ Cleaned dataset saved as fars2016_cleaned.csv and plots generated successfully! 🚦")
files.download('fars2016_cleaned.csv')
