#!/usr/bin/env python3
"""
MIND THE GAP — BART-AUGMENTED RECONSTRUCTION (FINAL NATIVE VERSION)

Data Structure Alignment:
  - Gap Marker : [MISSING] (Treated as standard subwords, no vocab resizing needed)
  - Target     : Full original text (Aligns with BART's pre-trained auto-encoder objective)
"""

import pandas as pd
import numpy as np
import torch
import re
import unicodedata
import inspect
from pathlib import Path
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
DATA_DIR      = ROOT / "data/processed"
MODEL_OUT     = ROOT / "models/bart_reconstruction_final"
BEST_DIR      = MODEL_OUT / "best_model"

TRAIN_CSV     = DATA_DIR / "mtg_reconstruction_train.csv"
VAL_CSV       = DATA_DIR / "mtg_reconstruction_val.csv"
TEST_CSV      = DATA_DIR / "mtg_reconstruction_test.csv"

VAL_PRED_CSV  = MODEL_OUT / "val_predictions.csv"
TEST_PRED_CSV = MODEL_OUT / "test_predictions.csv"
METRICS_PATH  = MODEL_OUT / "metrics_summary.txt"

# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME        = "facebook/bart-large"
GAP_TOKEN         = "[MISSING]"      
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 512 
SEED              = 42

rouge_metric = evaluate.load("rouge")

# ── text utilities ────────────────────────────────────────────────────────────
def normalize(x) -> str:
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x)).strip()
    return re.sub(r"\s+", " ", x)

def normalize_lower(x) -> str:
    return normalize(x).lower()

def token_overlap(pred: str, gold: str):
    def tok(t):
        return re.findall(r"\w+|[^\w\s]", normalize_lower(t), flags=re.UNICODE)
    p, g = tok(pred), tok(gold)
    if not p and not g: return 1.0, 1.0, 1.0
    if not p or not g:  return 0.0, 0.0, 0.0
    pc, gc = {}, {}
    for t in p: pc[t] = pc.get(t, 0) + 1
    for t in g: gc[t] = gc.get(t, 0) + 1
    ov = sum(min(c, gc.get(t, 0)) for t, c in pc.items())
    pr = ov / len(p);  re_ = ov / len(g)
    f1 = 0.0 if (pr + re_) == 0 else 2 * pr * re_ / (pr + re_)
    return pr, re_, f1

def full_metrics(preds, refs):
    exact, prec, rec, f1 = [], [], [], []
    for p, r in zip(preds, refs):
        exact.append(int(normalize_lower(p) == normalize_lower(r)))
        a, b, c = token_overlap(p, r)
        prec.append(a); rec.append(b); f1.append(c)
    out = {
        "accuracy":  float(np.mean(exact)),
        "precision": float(np.mean(prec)),
        "recall":    float(np.mean(rec)),
        "f1":        float(np.mean(f1)),
    }
    rouge = rouge_metric.compute(predictions=preds, references=refs)
    out.update(rouge)
    return out

# ── data preparation ──────────────────────────────────────────────────────────
def fix_adu_type(row) -> str:
    v = str(row.get("adu_type", "")).strip()
    if v and v.lower() not in ("nan", "none", ""):
        return v
    return "Claim" if str(row.get("input_text", "")).rstrip().endswith("[MISSING]") else "Premise"

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["id", "input_text", "target_text", "removed_sentence", "topic", "adu_type"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize)

    df["adu_type"] = df.apply(fix_adu_type, axis=1)

    # Build augmented input
    df["augmented_input"] = (
        "Topic: "       + df["topic"]    + " | "
        "Missing ADU: " + df["adu_type"] + " | "
        "Argument: "    + df["input_text"]
    )

    df = df[df["id"]          != ""].copy()
    df = df[df["input_text"]  != ""].copy()
    df = df[df["target_text"] != ""].copy()
    
    # Ensure [MISSING] is actually in the input
    df = df[df["augmented_input"].str.contains(re.escape(GAP_TOKEN), regex=True)].copy()
    df = df.drop_duplicates(subset=["id", "augmented_input", "target_text"])
    return df.reset_index(drop=True)

# ── tokenisation ──────────────────────────────────────────────────────────────
def make_preprocess(tokenizer):
    def preprocess(examples):
        # Dynamic Padding: Let the Collator handle -100 padding automatically
        model_inputs = tokenizer(
            examples["augmented_input"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        
        target_enc = tokenizer(
            text_target=examples["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
        
        model_inputs["labels"] = target_enc["input_ids"]
        return model_inputs
    return preprocess

# ── compute_metrics ───────────────────────────────────────────────────────────
def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        
        # 1. THE FIX: Replace -100 in BOTH preds and labels with pad_token_id
        preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        
        # 2. Decode safely
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 3. Clean strings
        decoded_preds  = [normalize(p) for p in decoded_preds]
        decoded_labels = [normalize(l) for l in decoded_labels]
        
        # 4. Calculate ROUGE
        rouge = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {k: round(v, 4) for k, v in rouge.items()
                if k in ("rouge1", "rouge2", "rougeL")}
    return compute_metrics

# ── training arguments ────────────────────────────────────────────────────────
def build_training_args():
    kwargs = dict(
        output_dir                  = str(MODEL_OUT),
        max_grad_norm               = 1.0, 
        warmup_steps                = 200,         
        
        # 1. HARD CAP THE LEARNING RATE
        learning_rate               = 4e-6,        
        
        # 2. INCREASE ADAM EPSILON
        adam_epsilon                = 1e-6,        
        
        # 3. DISABLE LABEL SMOOTHING
        label_smoothing_factor      = 0.0,         
        
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,           
        per_device_eval_batch_size  = 16,
        weight_decay                = 0.01,
        num_train_epochs            = 5,
        save_strategy               = "epoch",
        save_total_limit            = 2,
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LENGTH,
        fp16                        = False,
        bf16                        = False,
        load_best_model_at_end      = True,
        metric_for_best_model       = "rougeL",
        greater_is_better           = True,
        logging_steps               = 20,
        report_to                   = "none",
        seed                        = SEED,
        data_seed                   = SEED,
    )
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    return Seq2SeqTrainingArguments(**kwargs)

# ── inference ─────────────────────────────────────────────────────────────────
def generate_predictions(df, model, tokenizer, output_csv, split_name, device):
    model.eval()
    generated = []
    for i, row in df.iterrows():
        enc = tokenizer(
            row["augmented_input"],
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens      = MAX_TARGET_LENGTH,
                min_new_tokens      = 10,
                num_beams           = 4,
                length_penalty      = 1.0,
                no_repeat_ngram_size= 3,
                early_stopping      = True,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        generated.append(normalize(text))
        if i % 100 == 0:
            print(f"  {split_name}: {i}/{len(df)}")

    out_df = df.copy()
    out_df["prediction_text"] = generated
    metrics = full_metrics(generated, df["target_text"].astype(str).tolist())
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved → {output_csv}")

    print(f"\n  === {split_name.upper()} EXAMPLES ===")
    for _, row in out_df.head(3).iterrows():
        print(f"  GOLD FULL : {row['target_text'][:150]}...")
        print(f"  PRED FULL : {row['prediction_text'][:150]}...")
        print()
    return metrics

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MIND THE GAP — BART-AUGMENTED RECONSTRUCTION (FINAL NATIVE)")
    print("=" * 70)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("  ✓ Model loaded natively. No vocabulary resizing required.")
    print("  ✓ Target set to full paragraph (MAX_TARGET_LENGTH = 512).")

    print("\nLoading datasets …")
    train_df = prepare_df(pd.read_csv(TRAIN_CSV))
    val_df   = prepare_df(pd.read_csv(VAL_CSV))
    test_df  = prepare_df(pd.read_csv(TEST_CSV))
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    gap_count = train_df["augmented_input"].str.contains(
        re.escape(GAP_TOKEN), regex=True
    ).sum()
    print(f"\n  Rows with {GAP_TOKEN} in input: {gap_count}/{len(train_df)}")

    preprocess = make_preprocess(tokenizer)
    remove_cols = list(train_df.columns)

    tok_train = Dataset.from_pandas(train_df).map(
        preprocess, batched=True, remove_columns=remove_cols
    )
    tok_val = Dataset.from_pandas(val_df).map(
        preprocess, batched=True, remove_columns=remove_cols
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = build_training_args()

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = tok_train,
        eval_dataset     = tok_val,
        processing_class = tokenizer,
        data_collator    = data_collator,
        compute_metrics  = make_compute_metrics(tokenizer),
    )

    print("\nStarting training …")
    train_result = trainer.train()

    print(f"\nSaving best model → {BEST_DIR}")
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    print("\nGenerating predictions …")
    val_metrics  = generate_predictions(val_df, model, tokenizer, VAL_PRED_CSV,  "val",  device)
    test_metrics = generate_predictions(test_df, model, tokenizer, TEST_PRED_CSV, "test", device)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("TRAIN METRICS\n")
        for k, v in train_result.metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nVALIDATION METRICS\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
        f.write("\nTEST METRICS\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS (Evaluating Full Text Reconstruction)")
    print("=" * 70)
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {f'{v:.4f}' if isinstance(v, float) else v}")
    print("=" * 70)

if __name__ == "__main__":
    main()
