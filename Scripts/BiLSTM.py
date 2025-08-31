import os
import pandas as pd
import numpy as np
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
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# Download NLTK resources (silent mode)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ========== 1. DATA LOADING WITH ERROR HANDLING ==========
try:
    # Get the absolute path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_data.csv")
    
    # Verify file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}\n"
                              f"Please ensure 'sentiment_data.csv' is in the same directory as your script.")
    
    # Load and clean data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Comment', 'Sentiment'])  # Remove rows with missing values
    
    # Check required columns
    if not all(col in df.columns for col in ['Comment', 'Sentiment']):
        raise ValueError("CSV must contain both 'Comment' and 'Sentiment' columns")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ========== 2. IMPROVED TEXT PREPROCESSING ==========
stop_words = set(stopwords.words('english'))  # Preload for efficiency

def preprocess(text):
    # Handle missing values
    if pd.isna(text):
        return ""
    
    text = str(text).lower()  # Ensure string and lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation (faster than translate)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'].str.strip() != '']  # Remove empty comments

# ========== 3. TOKENIZATION AND PADDING ==========
MAX_WORDS = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# ========== 4. LABEL PROCESSING ==========
# Convert string labels to numerical if needed
if df['Sentiment'].dtype == 'object':
    label_map = {label: idx for idx, label in enumerate(df['Sentiment'].unique())}
    df['Sentiment'] = df['Sentiment'].map(label_map)

labels = to_categorical(df['Sentiment'])

# ========== 5. TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=df['Sentiment']  # Maintain class distribution
)

# ========== 6. BiLSTM MODEL ==========
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),  # Added regularization
    Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.build(input_shape=(None, MAX_LEN))  # <<< This builds the model
model.summary()

# ========== 7. TRAINING ==========
print("\nTraining model...")
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ========== 8. EVALUATION ==========
print("\nEvaluating model...")
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix with dynamic labels
label_names = list(label_map.keys()) if 'label_map' in locals() else ["Negative", "Neutral", "Positive"]
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_true, y_pred),
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("BiLSTM Sentiment Analysis Confusion Matrix")
plt.tight_layout()
plt.show()