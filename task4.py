# Task-04: Sentiment Analysis & Visualization
# Prodigy InfoTech Data Science Internship

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import zipfile, os
from google.colab import files

# --- Step 1: Load Dataset ---
zip_file_path = 'Twitter_Data.csv.zip'   # Upload this ZIP before running
extract_path = 'extracted_data'
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

csv_file_path = os.path.join(extract_path, 'Twitter_Data.csv')
df = pd.read_csv(csv_file_path, encoding='latin-1')
print("✅ Dataset Loaded Successfully")
print(df.head())

# --- Step 2: Data Cleaning ---
df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)

# --- Step 3: Sentiment Analysis using TextBlob ---
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment'] = df['clean_text'].apply(get_sentiment)
print("\nSentiment Counts:\n", df['Sentiment'].value_counts())

# --- Step 4: Visualization ---

# (a) Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=200)
plt.show()
files.download('sentiment_distribution.png')

# (b) Word Cloud Function
def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# WordClouds by sentiment
plot_wordcloud(' '.join(df[df['Sentiment']=='Positive']['clean_text']), "Word Cloud – Positive Tweets")
plot_wordcloud(' '.join(df[df['Sentiment']=='Negative']['clean_text']), "Word Cloud – Negative Tweets")
plot_wordcloud(' '.join(df[df['Sentiment']=='Neutral']['clean_text']),  "Word Cloud – Neutral Tweets")

# --- Step 5: Save Cleaned Data ---
df.to_csv('twitter_sentiment_cleaned.csv', index=False)
print("✅ Cleaned dataset saved as twitter_sentiment_cleaned.csv")
files.download('twitter_sentiment_cleaned.csv')
