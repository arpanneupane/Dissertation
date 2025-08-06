import os
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download required NLTK data
#nltk.download('punkt_tab')
#nltk.download('stopwords')

# Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sentiment_data.csv")
df = pd.read_csv(csv_path)
#df = pd.read_csv("sentiment_data.csv")

# Text preprocessing
#import pandas as pd
#import re

def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()  # Ensure text is string and lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

#df = pd.read_csv("/Users/arpan/Desktop/Dissertation/sentiment_data.csv")
df['Comment'] = df['Comment'].fillna('')  # Handle NaN values
df['cleaned'] = df['Comment'].apply(preprocess)

# Rest of your Naive Bayes code...

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Naive Bayes - Sentiment Analysis Confusion Matrix")
plt.show()
