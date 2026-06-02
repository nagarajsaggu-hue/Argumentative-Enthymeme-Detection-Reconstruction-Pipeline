#!/usr/bin/env python3
import pandas as pd
import random
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
INPUT_CSV = ROOT / "data/processed/mtg_step4b_final_enthymemes.csv"
OUTPUT_DIR = ROOT / "data/processed"
SEP_TOKEN = "[SEP]"

def is_locked_by_next(current_idx, all_sentences):
    """Fix for Doubt 4: Antecedent Lock (Rather/This/Therefore)"""
    if current_idx + 1 >= len(all_sentences): return False
    next_s = all_sentences[current_idx + 1].lower().strip()
    hooks = ['rather', 'instead', 'this', 'these', 'those', 'consequently', 'therefore', 'however', 'thus', 'hence']
    return any(next_s.startswith(h) for h in hooks)

def process_subset(df_subset, nlp):
    instances = []
    adu_counts = {"Premise": 0, "Claim": 0, "MajorClaim": 0}
    
    for _, row in df_subset.iterrows():
        doc = nlp(row['text'])
        orig_sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        
        # Doubt 2 Fix: Context Minimum
        if len(orig_sents) < 3 or len(row['text']) < 100: continue

        # Doubt 4 Fix: Guardrail
        rem_idx = int(row['removed_idx'])
        if is_locked_by_next(rem_idx, orig_sents): continue

        # AUDIT MAPPING (Match Stahl et al. 2023 distribution)
        # 1=Premise, 2=Claim, 3=MajorClaim
        label_map = {1: "Premise", 2: "Claim", 3: "MajorClaim"}
        adu_type = label_map.get(row['adu_label'], "Premise")
        adu_counts[adu_type] += 1

        enth_sents = [s for i, s in enumerate(orig_sents) if i != rem_idx]
        
        # (a) POSITIVE - Correct Position (Doubt 3: Edge cases handled)
        pos = list(enth_sents); pos.insert(rem_idx, SEP_TOKEN)
        instances.append({'input_text': " ".join(pos), 'label': 1, 'id': row['File']})

        # (b) NEGATIVE A - Wrong Position
        w_idx = random.choice([i for i in range(len(enth_sents) + 1) if i != rem_idx])
        neg_a = list(enth_sents); neg_a.insert(w_idx, SEP_TOKEN)
        instances.append({'input_text': " ".join(neg_a), 'label': 0, 'id': row['File']})

        # (c) NEGATIVE B - Robustness (Full argument)
        orig_gap = random.randint(0, len(orig_sents))
        neg_b = list(orig_sents); neg_b.insert(orig_gap, SEP_TOKEN)
        instances.append({'input_text': " ".join(neg_b), 'label': 0, 'id': row['File']})

    return pd.DataFrame(instances), adu_counts

def main():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(INPUT_CSV)
    random.seed(42)

    # LEAKAGE PROOF: Split by ID
    unique_ids = df['File'].unique().tolist()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=42)

    train_final, t_adu = process_subset(df[df['File'].isin(train_ids)], nlp)
    val_final, v_adu = process_subset(df[df['File'].isin(val_ids)], nlp)
    test_final, te_adu = process_subset(df[df['File'].isin(test_ids)], nlp)

    # Save
    train_final.to_csv(OUTPUT_DIR / "mtg_detection_train.csv", index=False)
    val_final.to_csv(OUTPUT_DIR / "mtg_detection_val.csv", index=False)
    test_final.to_csv(OUTPUT_DIR / "mtg_detection_test.csv", index=False)
     
    # Save combined overall file
    overall_final = pd.concat([train_final, val_final, test_final], ignore_index=True)
    overall_final.to_csv(OUTPUT_DIR / "mtg_detection_overall.csv", index=False)
    
    # FINAL AUDIT REPORT
    total = sum(t_adu.values()) + sum(v_adu.values()) + sum(te_adu.values())
    p = (t_adu['Premise'] + v_adu['Premise'] + te_adu['Premise']) / total * 100
    c = (t_adu['Claim'] + v_adu['Claim'] + te_adu['Claim']) / total * 100
    m = (t_adu['MajorClaim'] + v_adu['MajorClaim'] + te_adu['MajorClaim']) / total * 100

    print("\n" + "="*60)
    print("MIND THE GAP: FINAL AUDIT REPORT")
    print("="*60)
    print(f"Premise Removals:     {p:.2f}% (Target: 63.28%)")
    print(f"Claim Removals:       {c:.2f}% (Target: 30.97%)")
    print(f"MajorClaim Removals:  {m:.2f}% (Target: 5.75%)")
    print("-" * 60)
    print(f"Data Leakage Check:   {' PASSED' if not set(train_ids).intersection(test_ids) else '❌ FAILED'}")
    print(f"Overall file saved:   {OUTPUT_DIR / 'mtg_detection_overall.csv'}")
    print("="*60)

if __name__ == "__main__":
    main()
