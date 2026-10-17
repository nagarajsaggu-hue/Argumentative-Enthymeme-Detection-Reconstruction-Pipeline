#!/usr/bin/env python3
"""
Download and prepare Stab & Gurevych Argument Annotated Essays
for fine-tuning ADU classifier (4 labels: MajorClaim, Claim, Premise, None)
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ADU_DATA = ROOT / "data" / "adu_training"
ADU_DATA.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING STAB & GUREVYCH ARGUMENT ANNOTATED ESSAYS")
print("=" * 70)

# Load from HuggingFace datasets hub
print("\nLoading dataset from HuggingFace...")
dataset = load_dataset("ThuyNT/student_essay_argument_mining")
print(f"Dataset loaded: {dataset}")
print(f"Features: {dataset['train'].features}")

# Check label distribution
print("\nLabel distribution (train):")
from collections import Counter
labels = dataset['train']['label']
print(Counter(labels))

# Save splits
for split in dataset.keys():
    df = pd.DataFrame(dataset[split])
    out_path = ADU_DATA / f"{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {split}: {len(df)} rows → {out_path}")

print("\nDONE")
