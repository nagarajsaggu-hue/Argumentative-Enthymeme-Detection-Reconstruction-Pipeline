#!/usr/bin/env python3
"""
MIND THE GAP - STEP 4a: Optimized ADU Inference
Integrates Dynamic Padding with Step 3a Paragraph Logic.
Filters by: Naturalness (Step 3a), Length (>5 tokens), and ADU Type.
"""

import pandas as pd
import torch
import spacy
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding

# Setup Paths
ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
INPUT_CSV = ROOT / "data/processed/mtg_step3a_naturalness.csv"
OUTPUT_CSV = ROOT / "data/processed/mtg_step4a_adu_candidates.csv"
MODEL_DIR = ROOT / "models/adu_classifier_4class"

ID2LABEL = {0: "MajorClaim", 1: "Claim", 2: "Premise", 3: "None"}

class CandidateDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)

def main():
    print("=" * 70)
    print("MIND THE GAP - STEP 4a: UNIFIED ADU FILTERING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp = spacy.load("en_core_web_sm")
    
    print(f"Loading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    final_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Arguments"):
        para_text = row['text']
        # Load indices that passed Naturalness Filter (Step 3a)
        natural_indices = json.loads(row['natural_candidates'])
        
        doc = nlp(para_text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        
        # 1. Gather text for candidates that meet the Length Filter (>5 tokens)
        candidate_map = [] # List of (original_index, sentence_text)
        for i in natural_indices:
            if i < len(sentences):
                sent_text = sentences[i]
                if len(nlp(sent_text)) > 5:
                    candidate_map.append((i, sent_text))
        
        if not candidate_map:
            continue

        # 2. Batch Inference for this specific paragraph's candidates
        batch_texts = [item[1] for item in candidate_map]
        dataset = CandidateDataset(batch_texts, tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=len(batch_texts), collate_fn=data_collator)
        
        paragraph_adu_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                
                # 3. Filter for Argumentative Labels (0, 1, 2)
                for pred, (orig_idx, _) in zip(preds, candidate_map):
                    if pred in [0, 1, 2]:
                        paragraph_adu_indices.append(orig_idx)
        
        if paragraph_adu_indices:
            row_dict = row.to_dict()
            row_dict['final_candidates'] = json.dumps(paragraph_adu_indices)
            final_results.append(row_dict)

    # Save Results
    out_df = pd.DataFrame(final_results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Success! Saved {len(out_df)} valid arguments to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
