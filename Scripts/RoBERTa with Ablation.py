import os
import pandas as pd
import numpy as np
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.amp import autocast
import warnings
warnings.filterwarnings("ignore")

# ========== 1. DATA LOADING WITH ERROR HANDLING ==========
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sentiment_data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}\n"
                              f"Ensure 'sentiment_data.csv' is in the same directory.")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Comment', 'Sentiment'])

    if not all(col in df.columns for col in ['Comment', 'Sentiment']):
        raise ValueError("CSV must contain both 'Comment' and 'Sentiment' columns")

except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# ========== 2. TEXT PREPROCESSING ==========
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

df['cleaned'] = df['Comment'].apply(preprocess)
df = df[df['cleaned'] != '']

# ========== 3. DATA PREPARATION ==========
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

# ========== 4. TOKENIZATION AND DATASET ==========
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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

# ========== 5. EXPERIMENT FUNCTION ==========
def run_experiment(freeze_roberta: bool):
    print(f"\n{'='*30}\nExperiment: {'Frozen RoBERTa' if freeze_roberta else 'Full Fine-Tuning'}\n{'='*30}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get the number of unique labels from the training labels
    num_labels = len(set(train_labels))
    
    # Load RoBERTa model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels
    )
    
    if freeze_roberta:
        # Freeze all RoBERTa parameters except the classifier
        for param in model.roberta.parameters():
            param.requires_grad = False
    
    model.to(device)

    train_dataset = SentimentDataset(train_texts, train_labels)
    test_dataset = SentimentDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    # Training
    start_time = time.time()
    model.train()
    for epoch in range(3):  # RoBERTa typically benefits from more epochs than BERT
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type='mps' if device.type == 'mps' else 'cuda', dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    end_time = time.time()

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, preds))

    # Confusion matrix with numerical labels
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(true_labels, preds),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({'Frozen RoBERTa' if freeze_roberta else 'Full Model'})")
    plt.tight_layout()
    plt.show()

    print(f"\nTraining Time: {end_time - start_time:.2f} seconds")
    return acc, end_time - start_time, classification_report(true_labels, preds, output_dict=True)

# ========== 6. RUN BOTH EXPERIMENTS ==========
results = {}

# Run frozen RoBERTa experiment
acc_frozen, time_frozen, report_frozen = run_experiment(freeze_roberta=True)
results['Frozen RoBERTa'] = {'accuracy': acc_frozen, 'time': time_frozen, 'report': report_frozen}

# Run full fine-tuning experiment
acc_full, time_full, report_full = run_experiment(freeze_roberta=False)
results['Full Fine-Tuning'] = {'accuracy': acc_full, 'time': time_full, 'report': report_full}

# ========== 7. COMPARE RESULTS ==========
print("\n\n========= FINAL COMPARISON =========")
print(f"Frozen RoBERTa - Accuracy: {acc_frozen:.4f}, Training Time: {time_frozen:.2f}s")
print(f"Full Model - Accuracy: {acc_full:.4f}, Training Time: {time_full:.2f}s")

improvement = acc_full - acc_frozen
print(f"\nAccuracy Improvement from Freezing to Full: {improvement:.4f}")
print("Misclassification pattern differences can be further analyzed in the confusion matrices above.")

# Save results for further analysis
results_df = pd.DataFrame.from_dict(results, orient='index')
print("\nResults Summary:")
print(results_df[['accuracy', 'time']])