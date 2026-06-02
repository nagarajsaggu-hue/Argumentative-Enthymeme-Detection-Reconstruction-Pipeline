#!/usr/bin/env python3
"""
MIND THE GAP - STEP 5: RECONSTRUCTION INSTANCES (v2 Optimized)

Builds leakage-free reconstruction datasets from:
    data/processed/mtg_step4b_final_enthymemes.csv

CRITICAL UPDATES FOR BART STABILITY:
  1. MASK_TOKEN is now "[MISSING]" to avoid huggingface vocab-resizing bugs.
  2. target_text is now the FULL ORIGINAL PARAGRAPH, perfectly aligning 
     with BART's auto-encoder pre-training objective.
"""

import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import spacy


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")

INPUT_CSV = ROOT / "data/processed/mtg_step4b_final_enthymemes.csv"

OUTPUT_OVERALL_CSV = ROOT / "data/processed/mtg_reconstruction_overall.csv"
OUTPUT_TRAIN_CSV = ROOT / "data/processed/mtg_reconstruction_train.csv"
OUTPUT_VAL_CSV = ROOT / "data/processed/mtg_reconstruction_val.csv"
OUTPUT_TEST_CSV = ROOT / "data/processed/mtg_reconstruction_test.csv"


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# THE FIX: Use a standard string to avoid tokenizer resizing explosions
MASK_TOKEN = "[MISSING]"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def build_sentence_parser():
    return spacy.load("en_core_web_sm")


def split_sentences(text: str, nlp):
    text = normalize_text(text)
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def build_masked_text(full_text: str, removed_idx: int, nlp) -> str:
    sentences = split_sentences(full_text, nlp)

    if not sentences:
        raise ValueError("Could not split original text into sentences.")

    if removed_idx < 0 or removed_idx >= len(sentences):
        raise ValueError(
            f"Invalid removed_idx={removed_idx} for sentence count={len(sentences)}"
        )

    masked_sentences = sentences.copy()
    masked_sentences[removed_idx] = MASK_TOKEN
    masked_text = " ".join(masked_sentences).strip()

    if MASK_TOKEN not in masked_text:
        raise ValueError(f"Failed to insert {MASK_TOKEN} into input_text.")

    return masked_text


def is_valid_text(x) -> bool:
    x = normalize_text(x)
    return len(x) > 0


def prepare_reconstruction_row(row, nlp):
    sample_id = normalize_text(row["File"])
    full_text = normalize_text(row["text"])
    removed_sentence = normalize_text(row["removed_sentence"])
    removed_idx = int(row["removed_idx"])
    
    # Extract contextual metadata
    topic = normalize_text(row.get("Title", ""))
    
    # Exact ADU Mapping
    adu_mapping = {
        0: "MajorClaim",
        1: "Claim",
        2: "Premise",
        3: "None"
    }
    
    raw_adu = row.get("adu_label", "")
    try:
        if pd.isna(raw_adu) or str(raw_adu).strip() == "":
            adu_type = "Premise" # Safe fallback
        else:
            adu_int = int(float(raw_adu))
            adu_type = adu_mapping.get(adu_int, str(raw_adu))
    except ValueError:
        adu_type = normalize_text(raw_adu)

    input_text = build_masked_text(full_text, removed_idx, nlp)

    # THE SILVER BULLET: Target is the FULL original text.
    target_text = full_text

    return {
        "id": sample_id,
        "input_text": input_text,
        "target_text": target_text,
        "removed_sentence": removed_sentence, # Kept for precise ROUGE evaluation later
        "topic": topic,
        "adu_type": adu_type
    }


def leakage_free_split_by_id(df: pd.DataFrame):
    unique_ids = sorted(df["id"].astype(str).unique().tolist())
    random.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    train_df = df[df["id"].isin(train_ids)].copy()
    val_df = df[df["id"].isin(val_ids)].copy()
    test_df = df[df["id"].isin(test_ids)].copy()

    return train_df, val_df, test_df, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    set_seed(SEED)

    print("=" * 80)
    print("MIND THE GAP - STEP 5: RECONSTRUCTION INSTANCES (v2 Optimized)")
    print("=" * 80)
    print(f"Input file        : {INPUT_CSV}")
    print(f"Overall output    : {OUTPUT_OVERALL_CSV}")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUTPUT_OVERALL_CSV.parent.mkdir(parents=True, exist_ok=True)

    nlp = build_sentence_parser()
    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded rows: {len(df)}")

    required_columns = [
        "File", "Title", "text", "removed_sentence", "removed_idx", "adu_label"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = []
    skipped = 0
    seen_ids = set()

    for _, row in df.iterrows():
        try:
            # 1. Grab the ID first to check if we already processed this essay
            file_id = normalize_text(row["File"])
            
            if not file_id:
                skipped += 1
                continue
                
            # ONE-PER-ESSAY STRICT RULE
            if file_id in seen_ids:
                skipped += 1
                continue

            # 2. Proceed with normal validation
            if not is_valid_text(row["text"]):
                skipped += 1
                continue
            if not is_valid_text(row["removed_sentence"]):
                skipped += 1
                continue
            if pd.isna(row["removed_idx"]):
                skipped += 1
                continue

            rec = prepare_reconstruction_row(row, nlp)

            # Final sanity checks
            if not is_valid_text(rec["id"]):
                skipped += 1
                continue
            if not is_valid_text(rec["input_text"]):
                skipped += 1
                continue
            if not is_valid_text(rec["target_text"]):
                skipped += 1
                continue
            if MASK_TOKEN not in rec["input_text"]:
                skipped += 1
                continue

            # 3. Add to our dataset and mark this essay ID as 'seen'
            records.append(rec)
            seen_ids.add(file_id)

        except Exception as exc:
            skipped += 1
            print(f"Skipping row due to error: {exc}")

    recon_df = pd.DataFrame(records)

    if recon_df.empty:
        raise ValueError("No valid reconstruction instances were created.")

    before_dedup = len(recon_df)
    recon_df = recon_df.drop_duplicates(subset=["id"], keep="first").copy()
    
    # Sort for readability before split
    recon_df = recon_df.sort_values(["id"]).reset_index(drop=True)

    # Save overall
    recon_df.to_csv(OUTPUT_OVERALL_CSV, index=False)

    # Leakage-free split by id
    train_df, val_df, test_df, train_ids, val_ids, test_ids = leakage_free_split_by_id(recon_df)

    # Extra leakage safety checks
    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Data leakage detected across splits.")

    # Save splits
    train_df = train_df.sort_values(["id"]).reset_index(drop=True)
    val_df = val_df.sort_values(["id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["id"]).reset_index(drop=True)

    train_df.to_csv(OUTPUT_TRAIN_CSV, index=False)
    val_df.to_csv(OUTPUT_VAL_CSV, index=False)
    test_df.to_csv(OUTPUT_TEST_CSV, index=False)

    print("\n" + "=" * 80)
    print("RECONSTRUCTION INSTANCE GENERATION COMPLETED (STRICT 1-PER-ESSAY)")
    print("=" * 80)
    print(f"Original rows loaded          : {len(df)}")
    print(f"Valid reconstruction rows     : {before_dedup}")
    print(f"Skipped rows (Duplicates/Err) : {skipped}")
    print("-" * 80)
    print(f"Overall rows                  : {len(recon_df)}")
    print(f"Train rows                    : {len(train_df)}")
    print(f"Validation rows               : {len(val_df)}")
    print(f"Test rows                     : {len(test_df)}")
    print("=" * 80)

if __name__ == "__main__":
    main()