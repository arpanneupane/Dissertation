import os
import pandas as pd
import numpy as np
import string
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import seaborn as sns


 nltk.download('punkt', quiet=True)
 nltk.download('stopwords', quiet=True)

# ========== 1. DATA LOADING WITH ERROR HANDLING ==========
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_data.csv")
    
    # Verify file exists before loading
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"sentiment_data.csv not found at: {csv_path}")
    
    # Load and clean data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Comment', 'Sentiment'])  # Remove rows with missing values
    
    # Check if required columns exist
    if not all(col in df.columns for col in ['Comment', 'Sentiment']):
        raise ValueError("CSV must contain both 'Comment' and 'Sentiment' columns")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ========== 2. IMPROVED TEXT PREPROCESSING ==========
def preprocess(text):
    # Handle missing/NaN values
    if pd.isna(text):
        return ""
    
    # Convert to string in case of numeric input
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove punctuation (faster than str.translate)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'].str.strip() != '']  # Remove empty strings

# ========== 3. TOKENIZATION AND PADDING ==========
MAX_WORDS = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# ========== 4. PREPARE LABELS ==========
# Convert string labels to numerical if needed
if df['Sentiment'].dtype == 'object':
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['Sentiment'] = df['Sentiment'].map(sentiment_map)

labels = to_categorical(df['Sentiment'])

# ========== 5. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=df['Sentiment']  # Maintain class distribution
)

# ========== 6. LSTM MODEL ==========
model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Added dropout for regularization
    Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ========== 7. TRAINING ==========
history = model.fit(
    X_train, 
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ========== 8. EVALUATION ==========
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# Confusion matrix with auto-detected labels
label_names = ['negative', 'neutral', 'positive'] if 'Sentiment' in df.columns else ['0', '1', '2']
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Oranges',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("LSTM - Sentiment Analysis Confusion Matrix")
plt.show()
