#!/usr/bin/env python3
"""
MIND THE GAP - STEP 4b: EXACT PAPER RANKING
Formula: M = 0.5A + 0.5B
A: Semantic Centrality (sBERT)
B: Quality Reduction (Zero-Shot Coherence)
Uses public 'facebook/bart-large-mnli' as a proxy for Q.
"""

import pandas as pd
import numpy as np
import json
import spacy
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from tqdm import tqdm

# Setup Paths
ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
INPUT_CSV = ROOT / "data/processed/mtg_step4a_adu_candidates.csv"
OUTPUT_CSV = ROOT / "data/processed/mtg_step4b_final_enthymemes.csv"

def power_iteration(M, num_simulations: int = 100):
    n = M.shape[0]
    v = np.ones(n) / n
    for _ in range(num_simulations):
        v_next = M @ v
        v_next_norm = np.linalg.norm(v_next, ord=1)
        if v_next_norm == 0: break
        v = v_next / v_next_norm
    return v

def main():
    print("=" * 70)
    print("MIND THE GAP - STEP 4b: PAGERANK RANKING (PUBLIC MODELS)")
    print("=" * 70)

    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Load Resources
    print("Loading Centrality Model (sBERT: Public)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if device==0 else 'cpu')
    
    print("Loading Quality Proxy (BART-Large-MNLI: Public)...")
    # Using Zero-shot classification to score "Logical Strength"
    quality_pipe = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli", 
                            device=device)
    
    quality_label = "logically strong and coherent argument"
    
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(INPUT_CSV)
    final_data = []

    print(f"Ranking {len(df)} arguments...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        candidates = json.loads(row['final_candidates'])
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        n = len(sentences)
        
        if not candidates or n < 3: continue

        # --- MATRIX A: Centrality ---
        embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
        A_raw = util.cos_sim(embeddings, embeddings).cpu().numpy()
        A = A_raw / (A_raw.sum(axis=1, keepdims=True) + 1e-9)

        # --- MATRIX B: Quality Reduction ---
        # 1. Score original text
        res_orig = quality_pipe(text[:512], candidate_labels=[quality_label])
        q_orig = res_orig['scores'][0]
        
        quality_drops = np.zeros(n)
        for c_idx in candidates:
            # 2. Score text without this candidate
            text_without_c = " ".join([sentences[i] for i in range(n) if i != c_idx])
            res_rem = quality_pipe(text_without_c[:512], candidate_labels=[quality_label])
            q_removed = res_rem['scores'][0]
            
            # 3. Quality Reduction = Original Quality - Modified Quality
            quality_drops[c_idx] = max(0, q_orig - q_removed)
        
        # Build Matrix B
        B = np.tile(quality_drops, (n, 1))
        B_sum = B.sum(axis=1, keepdims=True)
        if B_sum.sum() > 0:
            B = B / (B_sum + 1e-9)
        else:
            B = np.zeros((n, n))

        # --- FINAL TRANSITION MATRIX: M = 0.5A + 0.5B ---
        M = 0.5 * A + 0.5 * B
        
        # PageRank Calculation
        scores = power_iteration(M)
        
        # Pick the best candidate based on highest PageRank influence
        candidate_scores = {c_idx: scores[c_idx] for c_idx in candidates if c_idx < n}
        if not candidate_scores: continue
        
        best_idx = max(candidate_scores, key=candidate_scores.get)
        
        res = row.to_dict()
        res['removed_sentence'] = sentences[best_idx]
        res['removed_idx'] = best_idx
        res['enthymeme_text'] = " ".join([sentences[i] for i in range(n) if i != best_idx])
        res['pagerank_score'] = float(candidate_scores[best_idx])
        final_data.append(res)

    # Save
    pd.DataFrame(final_data).to_csv(OUTPUT_CSV, index=False)
    print(f"\n CORPUS COMPLETE: {len(final_data)} enthymemes saved.")

if __name__ == "__main__":
    main()
