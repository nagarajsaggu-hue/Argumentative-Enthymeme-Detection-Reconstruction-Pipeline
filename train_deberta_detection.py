import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
MODEL_DIR = ROOT / "models/adu_classifier_4class"
TEST_DATA = ROOT / "data/adu_training/test_sentences.csv"
LABEL2ID = {"MajorClaim": 0, "Claim": 1, "Premise": 2, "None": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

df = pd.read_csv(TEST_DATA).dropna()
texts = df["text"].tolist()
true_labels = df["adu_label"].map(LABEL2ID).tolist()

class SimpleDataset(Dataset):
    def __init__(self, texts):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.enc.items()}

loader = DataLoader(SimpleDataset(texts), batch_size=32)
preds = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds.extend(torch.argmax(model(**batch).logits, dim=1).cpu().tolist())

print("\n--- SCIENTIFIC CLASSIFICATION REPORT ---")
print(classification_report(true_labels, preds, target_names=list(LABEL2ID.keys())))
