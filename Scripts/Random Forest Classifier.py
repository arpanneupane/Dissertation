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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt', quiet=True)  # Suppress download messages
nltk.download('stopwords', quiet=True)

# ====== 1. LOAD DATASET WITH ERROR HANDLING ======
try:
    # Dynamically locate CSV (works regardless of working directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_data.csv")
    
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    if 'Comment' not in df.columns or 'Sentiment' not in df.columns:
        raise KeyError("CSV must contain 'Comment' and 'Sentiment' columns.")
        
except FileNotFoundError:
    print(f"Error: File 'sentiment_data.csv' not found at {csv_path}")
    exit()
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit()

# ====== 2. IMPROVED TEXT PREPROCESSING ======
def preprocess(text):
    # Handle NaN/float values
    if pd.isna(text):
        return ""
    
    # Convert to string (in case of numeric input)
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing (drop rows with empty comments after cleaning)
df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'].str.strip() != '']  # Remove empty comments

# ====== 3. TF-IDF VECTORIZATION ======
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['Sentiment']

# ====== 4. TRAIN-TEST SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Preserve class distribution
)

# ====== 5. TRAIN RANDOM FOREST ======
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
model.fit(X_train, y_train)

# ====== 6. EVALUATION ======
y_pred = model.predict(X_test)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix with labels
labels = sorted(y.unique())  # Auto-detect sentiment classes (e.g., ['Neg', 'Neu', 'Pos'])
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Greens',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest - Sentiment Analysis Confusion Matrix")
plt.tight_layout()
plt.show()