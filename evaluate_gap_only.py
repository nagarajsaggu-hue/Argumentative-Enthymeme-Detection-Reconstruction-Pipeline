#!/usr/bin/env python3
import re
import pandas as pd
from pathlib import Path
import spacy

# Load spaCy for accurate sentence splitting
nlp = spacy.load("en_core_web_sm")

ROOT      = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
ANN_DIR   = ROOT / "data/adu_training/ArgumentAnnotatedEssays-2.0/brat-project-final"
OUT_DIR   = ROOT / "data/adu_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"MajorClaim": "MajorClaim", "Claim": "Claim", "Premise": "Premise"}

print("=" * 70)
print("PREPARING SENTENCE-LEVEL ADU DATA FROM LOCAL BRAT FILES")
print("=" * 70)

def process_local_files():
    records = []
    ann_files = list(ANN_DIR.glob("*.ann"))
    
    if not ann_files:
        print(f" ERROR: No .ann files found in {ANN_DIR}")
        return pd.DataFrame()

    for ann_path in ann_files:
        txt_path = ann_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
            
        full_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        
        # 1. Extract raw ADU spans from .ann file
        annotations = []
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("T"): continue
            parts = line.split("\t")
            meta  = parts[1].split()
            label, start, end = meta[0], int(meta[1]), int(meta[2])
            if label in LABEL_MAP:
                annotations.append({'start': start, 'end': end, 'label': label})
        
        # 2. Split the essay into full sentences using spaCy
        doc = nlp(full_text)
        for sent in doc.sents:
            s_text = sent.text.strip()
            s_start = sent.start_char
            s_end = sent.end_char
            
            # Skip sentences with 5 or fewer tokens (Paper requirement)
            if len(s_text.split()) <= 5:
                continue
                
            # 3. Map annotation to the full sentence
            # We check if an ADU span overlaps with this sentence
            final_label = "None"
            best_overlap = 0
            for anno in annotations:
                overlap = max(0, min(s_end, anno['end']) - max(s_start, anno['start']))
                if overlap > best_overlap:
                    best_overlap = overlap
                    final_label = anno['label']
            
            records.append({"text": s_text, "adu_label": final_label})
            
    return pd.DataFrame(records)

print("Processing files...")
df_all = process_local_files()

if not df_all.empty:
    # Shuffle and split 70/10/20 per paper logic
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(0.7 * len(df_all))
    val_end = int(0.8 * len(df_all))
    
    train_df = df_all[:train_end]
    val_df   = df_all[train_end:val_end]
    test_df  = df_all[val_end:]
    
    for name, df_split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        out_path = OUT_DIR / f"{name}_sentences.csv"
        df_split.to_csv(out_path, index=False)
        print(f"\nSaved {name}: {len(df_split)} rows")
        print(df_split['adu_label'].value_counts())

print("\n LOCAL DATA PREPARATION COMPLETE")
