#!/usr/bin/env python3
"""
MIND THE GAP — FINAL EVALUATION (Gap-Only Extraction)
Calculates ROUGE scores specifically for the generated text, 
broken down by gap position (Prefix, Suffix, Middle).
"""

import pandas as pd
import evaluate
import re
import warnings

# Suppress evaluate warnings for cleaner logs
warnings.filterwarnings("ignore")

PREDICTIONS_CSV = "/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project/models/bart_reconstruction_final/test_predictions.csv"
MASK_TOKEN = "[MISSING]"

def extract_generated_gap(input_masked: str, full_prediction: str) -> str:
    """Isolates the text BART generated to replace [MISSING]."""
    parts = input_masked.split(MASK_TOKEN)
    if len(parts) != 2:
        return "" 
        
    prefix = parts[0].split("Argument: ")[-1].strip()
    suffix = parts[1].strip()
    
    pred_text = full_prediction.strip()
    
    # Strip prefix
    if prefix and pred_text.startswith(prefix):
        pred_text = pred_text[len(prefix):].strip()
    elif prefix:
        pred_text = pred_text.replace(prefix, "").strip()
        
    # Strip suffix
    if suffix and pred_text.endswith(suffix):
        pred_text = pred_text[:-len(suffix)].strip()
    elif suffix:
        pred_text = pred_text.replace(suffix, "").strip()
        
    return pred_text

def determine_position(input_masked: str) -> str:
    """Determines if the gap is at the start, end, or middle."""
    arg_text = input_masked.split("Argument: ")[-1].strip()
    if arg_text.startswith(MASK_TOKEN):
        return "Prefix (Cold Start)"
    elif arg_text.endswith(MASK_TOKEN):
        return "Suffix (Cliffhanger)"
    else:
        return "Middle (Interpolation)"

def main():
    print("Loading predictions from:")
    print(PREDICTIONS_CSV)
    df = pd.read_csv(PREDICTIONS_CSV)
    
    # 1. Extract the generated gaps and positions
    df['extracted_prediction'] = df.apply(lambda row: extract_generated_gap(row['augmented_input'], row['prediction_text']), axis=1)
    df['gap_position'] = df['augmented_input'].apply(determine_position)
    
    # Clean up empty generations to avoid evaluation crashes
    df['extracted_prediction'] = df['extracted_prediction'].fillna("").replace("", "EMPTY")
    df['removed_sentence'] = df['removed_sentence'].fillna("").astype(str)
    
    rouge_metric = evaluate.load("rouge")
    
    print("\n" + "="*60)
    print("GAP-ONLY EVALUATION RESULTS")
    print("="*60)
    
    # 2. Overall Score
    overall_rouge = rouge_metric.compute(
        predictions=df['extracted_prediction'].tolist(),
        references=df['removed_sentence'].tolist(),
        use_stemmer=True
    )
    print(f"OVERALL (N={len(df)})")
    print(f"  ROUGE-1: {overall_rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {overall_rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {overall_rouge['rougeL']:.4f}")
    print("-" * 60)
    
    # 3. Breakdown by Position
    for position in ["Middle (Interpolation)", "Prefix (Cold Start)", "Suffix (Cliffhanger)"]:
        sub_df = df[df['gap_position'] == position]
        if len(sub_df) == 0: continue
            
        sub_rouge = rouge_metric.compute(
            predictions=sub_df['extracted_prediction'].tolist(),
            references=sub_df['removed_sentence'].tolist(),
            use_stemmer=True
        )
        print(f"{position} (N={len(sub_df)})")
        print(f"  ROUGE-1: {sub_rouge['rouge1']:.4f}")
        print(f"  ROUGE-2: {sub_rouge['rouge2']:.4f}")
        print(f"  ROUGE-L: {sub_rouge['rougeL']:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
