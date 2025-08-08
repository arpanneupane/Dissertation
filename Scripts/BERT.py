import os
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
from tqdm.auto import tqdm
from torch.amp import autocast

# ========== 1. DATA LOADING WITH ERROR HANDLING ==========
try:
    # Get the absolute path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_data.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}\n"
                              f"Please ensure 'sentiment_data.csv' is in the same directory as your script.")
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Comment', 'Sentiment'])  # Remove rows with missing values
    
    if not all(col in df.columns for col in ['Comment', 'Sentiment']):
        raise ValueError("CSV must contain both 'Comment' and 'Sentiment' columns")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ========== 2. TEXT PREPROCESSING ==========
def preprocess(text):
    text = str(text).lower()  # Ensure string conversion
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special chars
    return text.strip()  # Remove extra whitespace

df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'] != '']  # Remove empty texts

# ========== 3. DATA PREPARATION ==========
# Convert labels to numerical if needed
if df['Sentiment'].dtype == 'object':
    label_map = {label: idx for idx, label in enumerate(sorted(df['Sentiment'].unique()))}
    df['Sentiment'] = df['Sentiment'].map(label_map)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned'].tolist(), 
    df['Sentiment'].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df['Sentiment']
)

# ========== 4. BERT TOKENIZATION ==========
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ========== 5. MODEL SETUP ==========
#Apple Silicon Chip
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=len(label_map) if 'label_map' in locals() else 3
)
model.to(device)

# ========== 6. TRAINING ==========
optimizer = AdamW(model.parameters(), lr=2e-5)

print("\nStarting training...")
model.train()
for epoch in range(1):  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # Mixed precision training
        with autocast(device_type='mps', dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

# ========== 7. EVALUATION ==========
print("\nEvaluating model...")
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

# ========== 8. RESULTS ==========
print("\nClassification Report:")
print(classification_report(true_labels, preds))

# Confusion matrix with dynamic labels
label_names = list(label_map.keys()) if 'label_map' in locals() else ["Negative", "Neutral", "Positive"]
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(true_labels, preds),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("BERT Sentiment Analysis Confusion Matrix")
plt.tight_layout()
plt.show()
