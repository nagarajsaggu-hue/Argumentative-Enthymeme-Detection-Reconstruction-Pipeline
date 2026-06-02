#!/usr/bin/env python3
"""
MIND THE GAP - STEP 1: Load dataset and compute sentence counts
Input:  data/raw/mind_the_gap_dataset.csv
Output: data/processed/mtg_step1_loaded.csv
"""

import pandas as pd
import spacy
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RAW      = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MIND THE GAP - STEP 1: LOAD DATASET")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load raw dataset
raw_path = RAW / "mind_the_gap_dataset.csv"
print(f"\nLoading: {raw_path}")

df = pd.read_csv(raw_path)
print(f"Columns found: {df.columns.tolist()}")
print(f"Total rows loaded: {len(df)}")

# Rename paragraph column
df = df.rename(columns={"Paragraph": "text"})

# Drop empty rows
before = len(df)
df = df.dropna(subset=["text"]).reset_index(drop=True)
print(f"Dropped empty rows: {before - len(df)}")

# Load spaCy
print("\nLoading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Sentence counting via batched pipe
print("Counting sentences (batched spaCy pipe)...")
sent_counts = []
for doc in nlp.pipe(df["text"].tolist(), batch_size=256, disable=["ner", "tagger"]):
    sent_counts.append(len(list(doc.sents)))
df["num_sents"] = sent_counts

# Word count per paragraph
df["text_length"] = df["text"].apply(lambda x: len(x.split()))

print("\nSentence count distribution:")
print(df["num_sents"].value_counts().sort_index().head(20).to_string())

print("\nText length stats (word count):")
print(df["text_length"].describe().to_string())

# Save
out_path = PROCESSED / "mtg_step1_loaded.csv"
df.to_csv(out_path, index=False)

# Report
report = f"""
MIND THE GAP - STEP 1: LOAD REPORT
====================================

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input:  {raw_path}
Output: {out_path}

DATASET STATS:
  Total rows loaded:     {len(df)}
  Columns:               {df.columns.tolist()}

SENTENCE COUNT DISTRIBUTION:
{df["num_sents"].value_counts().sort_index().head(20).to_string()}

TEXT LENGTH STATS (words):
{df["text_length"].describe().to_string()}

STEP 1 COMPLETE
"""

report_path = PROCESSED / "mtg_step1_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f"\n{'=' * 70}")
print(f"STEP 1 COMPLETE - Saved to {out_path}")
print(f"Report: {report_path}")
