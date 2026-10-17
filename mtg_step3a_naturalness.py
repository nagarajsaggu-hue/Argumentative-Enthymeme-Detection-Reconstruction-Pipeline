#!/usr/bin/env python3
"""
MIND THE GAP - STEP 4b Audit
Validates the statistical quality of the final enthymeme corpus.
"""
import pandas as pd
import json
import numpy as np

OUTPUT_FILE = "data/processed/mtg_step4b_final_enthymemes.csv"

def audit():
    print("="*70)
    print("CORPUS AUDIT: STEP 4b FINAL RESULTS")
    print("="*70)
    
    df = pd.read_csv(OUTPUT_FILE)
    
    # 1. Integrity Checks
    print(f"Total Enthymemes Generated: {len(df)}")
    
    # Check if removed_idx is always in final_candidates
    df['is_valid_selection'] = df.apply(
        lambda x: x['removed_idx'] in json.loads(x['final_candidates']), axis=1
    )
    invalid_count = len(df) - df['is_valid_selection'].sum()
    print(f"Selection Integrity: {'✅ PASS' if invalid_count == 0 else '❌ FAIL'} ({invalid_count} errors)")

    # 2. PageRank Score Distribution
    avg_score = df['pagerank_score'].mean()
    std_score = df['pagerank_score'].std()
    print(f"Average PageRank Score: {avg_score:.4f} (±{std_score:.4f})")

    # 3. Text Reduction Analysis
    # Ensure enthymemes are shorter than original texts
    df['original_len'] = df['text'].str.len()
    df['enthymeme_len'] = df['enthymeme_text'].str.len()
    avg_reduction = (df['original_len'] - df['enthymeme_len']).mean()
    print(f"Average Character Reduction: {avg_reduction:.1f} chars per essay")

    # 4. ADU Type Distribution (If adu_label is present)
    if 'adu_label' in df.columns:
        print("\nDistribution of Removed ADU Types:")
        dist = df['adu_label'].value_counts(normalize=True) * 100
        for label, pct in dist.items():
            print(f"  - {label:<12}: {pct:.1f}%")

    # 5. Cohesion Safety Check
    # Verify no enthymeme starts with a transition word like "Rather" or "Therefore"
    danger_words = ['rather', 'therefore', 'consequently', 'this', 'however']
    df['starts_with_danger'] = df['enthymeme_text'].str.lower().str.strip().apply(
        lambda x: any(x.startswith(w) for w in danger_words)
    )
    danger_count = df['starts_with_danger'].sum()
    print(f"\nDiscourse Guardrail Check: {danger_count} potential orphaned transitions found.")
    
    if danger_count > 0:
        print("Note: These are likely internal transitions that are now paragraph-initial.")

    print("="*70)

if __name__ == "__main__":
    audit()
