#!/usr/bin/env python3
"""
ADU CLASSIFIER - FINE-TUNING SCRIPT
Optimized with Dynamic Padding and Macro F1-Score Tracking.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup, DataCollatorWithPadding
)

ROOT      = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
DATA_DIR  = ROOT / "data/adu_training"
MODEL_OUT = ROOT / "models/adu_classifier_4class"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"MajorClaim": 0, "Claim": 1, "Premise": 2, "None": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_adu_df(name):
    path = DATA_DIR / f"{name}_sentences.csv"
    df = pd.read_csv(path)
    df['adu_label'] = df['adu_label'].fillna("None").astype(str)
    
    # ← ADD THIS LINE:
    df['text'] = df['text'].str.split('\n\n').str[-1].str.strip()
    
    df = df[df['adu_label'].isin(LABEL2ID.keys())].reset_index(drop=True)
    return df

print("Loading sentence-level CSVs and validating labels...")
train_df = load_adu_df("train")
val_df   = load_adu_df("validation")

class_counts = train_df['adu_label'].map(LABEL2ID).value_counts().sort_index().values
weights = 1.0 / class_counts
weights = weights / weights.sum()
tensor_weights = torch.tensor(weights, dtype=torch.float)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ADUDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["adu_label"].map(LABEL2ID).astype(int).tolist()
        
    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = tokenizer(self.texts[idx], truncation=True, max_length=128)
        item["labels"] = self.labels[idx]
        return item

train_ds = ADUDataset(train_df)
val_ds   = ADUDataset(val_df)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=data_collator)
val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID
).to(device)

loss_fn = nn.CrossEntropyLoss(weight=tensor_weights.to(device))

EPOCHS = 4
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dl) * EPOCHS)

best_val_loss = float("inf")
print(f"Starting training on {device}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = 0
    
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        lbls = batch.pop("labels")
        
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, lbls)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        t_loss += loss.item()
    
    model.eval()
    v_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbls = batch.pop("labels")
            outputs = model(**batch)
            
            loss = loss_fn(outputs.logits, lbls)
            v_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            
    avg_t = t_loss / len(train_dl)
    avg_v = v_loss / len(val_dl)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    print(f"Epoch {epoch:02d} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f} | Macro F1: {macro_f1:.4f}")
    
    if avg_v < best_val_loss:
        best_val_loss = avg_v
        model.save_pretrained(MODEL_OUT)
        tokenizer.save_pretrained(MODEL_OUT)
        print(f"  -> Model improved! Saved to {MODEL_OUT}")

print(f" SUCCESS: Sentence-level model training complete.")
