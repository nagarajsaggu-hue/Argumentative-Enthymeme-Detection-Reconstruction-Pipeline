#!/usr/bin/env python3
"""
MIND THE GAP - STEP 2: Filter arguments to 4+ sentences
Input:  data/processed/mtg_step1_loaded.csv
Output: data/processed/mtg_step2_filtered_4plus.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

print("=" * 70)
print("MIND THE GAP - STEP 2: FILTER TO 4+ SENTENCES")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv(PROCESSED / "mtg_step1_loaded.csv")
print(f"\nInput arguments: {len(df)}")

# Filter 1: Keep only 4+ sentences
df_filtered = df[df["num_sents"] >= 4].copy()

print(f"\n  Kept (>=4 sentences):    {len(df_filtered)}")
print(f"  Removed (<4 sentences):  {len(df) - len(df_filtered)}")
print(f"  Retention rate:          {len(df_filtered)/len(df)*100:.2f}%")

print("\nSentence distribution AFTER filtering:")
print(df_filtered["num_sents"].value_counts().sort_index().head(20).to_string())

# Save
out_path = PROCESSED / "mtg_step2_filtered_4plus.csv"
df_filtered.to_csv(out_path, index=False)

# Report
report = f"""
MIND THE GAP - STEP 2: FILTER REPORT
======================================

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input:  {PROCESSED / "mtg_step1_loaded.csv"}
Output: {out_path}

FILTERING RESULTS:
  Input arguments:         {len(df)}
  Kept (>=4 sentences):    {len(df_filtered)}
  Removed (<4 sentences):  {len(df) - len(df_filtered)}
  Retention rate:          {len(df_filtered)/len(df)*100:.2f}%

SENTENCE DISTRIBUTION (AFTER):
{df_filtered["num_sents"].value_counts().sort_index().head(20).to_string()}

STEP 2 COMPLETE
"""

report_path = PROCESSED / "mtg_step2_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f"\n{'=' * 70}")
print(f"STEP 2 COMPLETE - Saved to {out_path}")
print(f"Report: {report_path}")
