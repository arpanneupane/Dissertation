# **Code for Classical Models with Tuning (classical_models_tuning.py):**
# ```python
import os
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sentiment_data.csv")
df = pd.read_csv(csv_path)
# Text preprocessing
def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'].str.strip() != '']
# TF-IDF with unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['Sentiment']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Define models and hyperparameters for tuning
models = {
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    }
}
# Train and tune each model
for model_name, model_info in models.items():
    print(f"\n=== Tuning {model_name} ===")
    clf = GridSearchCV(model_info['model'], model_info['params'], cv=3, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print(f"Best parameters: {clf.best_params_}")
    print(f"Best macro-F1: {clf.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.show()
    
    # Save the model
    joblib.dump(clf.best_estimator_, f"{model_name.replace(' ', '_')}_model.pkl")
# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")