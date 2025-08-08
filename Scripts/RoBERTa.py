import os
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Data Loading and Preprocessing
def load_data(filepath):
    try:
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "sentiment_data.csv")
        df = pd.read_csv(csv_path)
        if not {'Comment', 'Sentiment'}.issubset(df.columns):
            raise ValueError("CSV must contain 'Comment' and 'Sentiment' columns")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit()

# Enhanced text cleaning
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    return text

# 2. Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 3. Main Function
def main():
    # Load and prepare data
    df = load_data("sentiment_data.csv")
    df['cleaned'] = df['Comment'].apply(clean_text)
    df = df[df['cleaned'] != ''].dropna(subset=['Sentiment'])
    
    # Convert labels if needed
    if df['Sentiment'].dtype == 'object':
        label_map = {label: i for i, label in enumerate(df['Sentiment'].unique())}
        df['Sentiment'] = df['Sentiment'].map(label_map)
    else:
        label_map = None
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned'].values,
        df['Sentiment'].values,
        test_size=0.2,
        random_state=SEED,
        stratify=df['Sentiment']
    )
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    num_labels = len(label_map) if label_map else len(df['Sentiment'].unique())
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Device setup (MPS for Apple Silicon Chip)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Average loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    preds, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    # Results
    print("\nClassification Report:")
    print(classification_report(true_labels, preds))
    
    # Confusion matrix
    label_names = list(label_map.keys()) if label_map else ["Negative", "Neutral", "Positive"]
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
    plt.title("RoBERTa Sentiment Analysis")
    plt.tight_layout()
    plt.show()
    
    # Save model
    model.save_pretrained("./roberta_sentiment_model")
    tokenizer.save_pretrained("./roberta_sentiment_model")
    print("\nModel saved to 'roberta_sentiment_model' directory")

if __name__ == "__main__":
    main()
    
