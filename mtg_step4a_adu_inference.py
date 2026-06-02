#!/usr/bin/env python3
"""
MIND THE GAP - STEP 3a: Naturalness ADU Filtering
Input:  data/processed/mtg_step2_filtered_4plus.csv
Output: data/processed/mtg_step3a_naturalness.csv
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
import spacy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

print("=" * 70)
print("MIND THE GAP - STEP 3a: NATURALNESS ADU FILTERING (HYBRID)")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Setup Device & Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading BERT NSP model and SpaCy...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(device)
model.eval()

nlp = spacy.load("en_core_web_sm")

# Load Data
in_path = PROCESSED / "mtg_step2_filtered_4plus.csv"
df = pd.read_csv(in_path)
print(f"\nInput arguments to process: {len(df)}")

def get_natural_candidates(text):
    """
    Returns a list of sentence indices that can be safely removed.
    Uses Rule-Based Heuristics FIRST, then BERT NSP.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    n = len(sentences)
    
    if n < 4:
        return []

    candidates = []
    
    # Explicit dependencies found in learner essays
    danger_markers = [
        "rather", "however", "instead", "consequently", "therefore", 
        "as a result", "furthermore", "moreover", "additionally", "in addition",
        "for example", "for instance", "as a consequence", "hence", "thus",
        "this", "these", "that", "those", "such", "on the other hand"
    ]
    
    for i in range(n):
        # The first and last sentences can always be removed without breaking a "bridge"
        if i == 0 or i == n - 1:
            candidates.append(i)
            continue
            
        seq_A = sentences[i-1]
        seq_B = sentences[i+1]
        
        # 1. HEURISTIC GUARDRAIL: Check if S_{i+1} depends on S_i
        seq_B_lower = seq_B.lower()
        is_dependent = any(
            seq_B_lower.startswith(marker + " ") or 
            seq_B_lower.startswith(marker + ",") 
            for marker in danger_markers
        )
        
        if is_dependent:
            continue # Skip immediately, do not run through BERT
            
        # 2. BERT NSP CHECK
        encoding = tokenizer(seq_A, seq_B, return_tensors='pt', truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # label 0 = IsNextSentence
            is_next = torch.argmax(probs).item() == 0
            
        if is_next:
            candidates.append(i)
            
    return candidates

# Process with progress bar
tqdm.pandas(desc="Filtering for Naturalness")
df['natural_candidates'] = df['text'].progress_apply(lambda x: json.dumps(get_natural_candidates(x)))

# Keep only rows that have at least one valid sentence we can remove
df_filtered = df[df['natural_candidates'] != "[]"].copy()

print(f"\nArguments with >=1 natural candidate: {len(df_filtered)}")
print(f"Arguments removed (no safe candidates): {len(df) - len(df_filtered)}")

# Save
out_path = PROCESSED / "mtg_step3a_naturalness.csv"
df_filtered.to_csv(out_path, index=False)

print(f"\n{'=' * 70}")
print(f"STEP 3a COMPLETE - Saved to {out_path}")
