#!/usr/bin/env python3
import os
import inspect
from pathlib import Path

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

ROOT = Path("/mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project")
TRAIN_CSV = ROOT / "data/processed/mtg_detection_train.csv"
VAL_CSV = ROOT / "data/processed/mtg_detection_val.csv"
TEST_CSV = ROOT / "data/processed/mtg_detection_test.csv"

MODEL_NAME = "microsoft/deberta-base"
OUTPUT_DIR = ROOT / "models/deberta_detection_threshold"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"

VAL_PRED_CSV = OUTPUT_DIR / "val_predictions.csv"
TEST_PRED_CSV = OUTPUT_DIR / "test_predictions.csv"
TEST_PRED_THRESH_CSV = OUTPUT_DIR / "test_predictions_thresholded.csv"
BEST_POS_PER_ID_CSV = OUTPUT_DIR / "test_best_position_per_id.csv"
METRICS_PATH = OUTPUT_DIR / "metrics_summary.txt"

MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 24
SEED = 42
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def compute_binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def split_on_sep(text):
    if pd.isna(text):
        return "", ""

    text = str(text)
    if "[SEP]" not in text:
        return text.strip(), ""

    left, right = text.split("[SEP]", 1)
    return left.strip(), right.strip()


def prepare_dataframe(df):
    left_texts = []
    right_texts = []

    for text in df["input_text"].tolist():
        left, right = split_on_sep(text)
        left_texts.append(left)
        right_texts.append(right)

    out_df = pd.DataFrame({
        "id": df["id"].astype(str).tolist(),
        "input_text": df["input_text"].astype(str).tolist(),
        "text_left": left_texts,
        "text_right": right_texts,
        "label": df["label"].astype(int).tolist(),
    })
    return out_df


def build_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset_dict, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text_left"],
            batch["text_right"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset_dict.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is None:
            loss = outputs.get("loss")
        else:
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def build_training_args():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=False,
        bf16=False,
        seed=SEED,
        data_seed=SEED,
        max_grad_norm=1.0,
    )

    sig = inspect.signature(TrainingArguments.__init__)

    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"

    if "save_total_limit" in sig.parameters:
        kwargs["save_total_limit"] = 2

    if "logging_strategy" in sig.parameters:
        kwargs["logging_strategy"] = "steps"

    if "warmup_ratio" in sig.parameters:
        kwargs["warmup_ratio"] = WARMUP_RATIO
    elif "warmup_steps" in sig.parameters:
        kwargs["warmup_steps"] = 0

    return TrainingArguments(**kwargs)


def probs_from_logits(logits):
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return probs[:, 1]


def tune_threshold(y_true, prob_1_values):
    best_threshold = 0.50
    best_metrics = None
    best_f1 = -1.0

    candidate_thresholds = np.arange(0.10, 0.91, 0.01)

    for thr in candidate_thresholds:
        preds = (prob_1_values >= thr).astype(int)
        metrics = compute_binary_metrics(y_true, preds)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def save_prediction_csv(df, prob_1_values, pred_labels, out_path):
    out_df = df.copy()
    out_df["true_label"] = out_df["label"].astype(int)
    out_df["pred_label"] = pred_labels.astype(int)
    out_df["prob_1"] = prob_1_values
    out_df["prob_0"] = 1.0 - out_df["prob_1"]

    out_df = out_df[
        ["id", "input_text", "true_label", "pred_label", "prob_0", "prob_1"]
    ]
    out_df.to_csv(out_path, index=False)


def save_best_position_per_id(df, prob_1_values, out_path):
    tmp = df.copy()
    tmp["prob_1"] = prob_1_values
    tmp["true_label"] = tmp["label"].astype(int)

    idx = tmp.groupby("id")["prob_1"].idxmax()
    best_df = tmp.loc[idx].copy()
    best_df["chosen_as_best_for_id"] = 1
    best_df["pred_label"] = 1

    best_df = best_df.sort_values(["id"]).reset_index(drop=True)
    best_df = best_df[
        ["id", "input_text", "true_label", "pred_label", "prob_1", "chosen_as_best_for_id"]
    ]
    best_df.to_csv(out_path, index=False)


def main():
    set_seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    train_df_raw = pd.read_csv(TRAIN_CSV)
    val_df_raw = pd.read_csv(VAL_CSV)
    test_df_raw = pd.read_csv(TEST_CSV)

    train_df = prepare_dataframe(train_df_raw)
    val_df = prepare_dataframe(val_df_raw)
    test_df = prepare_dataframe(test_df_raw)

    neg = int((train_df["label"] == 0).sum())
    pos = int((train_df["label"] == 1).sum())
    total = len(train_df)

    class_weights = torch.tensor(
        [
            total / (2.0 * max(neg, 1)),
            total / (2.0 * max(pos, 1)),
        ],
        dtype=torch.float
    )

    print("=" * 80)
    print("THRESHOLD-TUNED PAPER-ALIGNED ENTHYMEME DETECTION TRAINING")
    print("=" * 80)
    print(f"Model           : {MODEL_NAME}")
    print(f"Train file      : {TRAIN_CSV}")
    print(f"Validation file : {VAL_CSV}")
    print(f"Test file       : {TEST_CSV}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Best model dir  : {BEST_MODEL_DIR}")
    print(f"Val pred CSV    : {VAL_PRED_CSV}")
    print(f"Test pred CSV   : {TEST_PRED_CSV}")
    print(f"Test thr CSV    : {TEST_PRED_THRESH_CSV}")
    print(f"Best per id CSV : {BEST_POS_PER_ID_CSV}")
    print(f"Max length      : {MAX_LENGTH}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(f"Weight decay    : {WEIGHT_DECAY}")
    print(f"Warmup ratio    : {WARMUP_RATIO}")
    print("-" * 80)
    print("Train label distribution:")
    print(train_df["label"].value_counts().sort_index())
    print(f"Class weights   : {class_weights.tolist()}")
    print("-" * 80)
    print("Sample split example:")
    print("LEFT :", repr(train_df.iloc[0]["text_left"][:200]))
    print("RIGHT:", repr(train_df.iloc[0]["text_right"][:200]))
    print("=" * 80)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })

    tokenizer = build_tokenizer()
    tokenized = tokenize_dataset(ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.config.problem_type = "single_label_classification"

    training_args = build_training_args()

    callbacks = []
    try:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        )
    except TypeError:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("STANDARD VALIDATION EVALUATION")
    print("=" * 80)
    val_metrics = trainer.evaluate(tokenized["validation"])
    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("STANDARD TEST EVALUATION")
    print("=" * 80)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    # Validation predictions
    val_output = trainer.predict(tokenized["validation"])
    val_logits = val_output.predictions
    val_prob_1 = probs_from_logits(val_logits)
    val_true = val_df["label"].astype(int).values
    val_pred_default = (val_prob_1 >= 0.50).astype(int)

    save_prediction_csv(val_df, val_prob_1, val_pred_default, VAL_PRED_CSV)

    # Tune threshold on validation set
    best_threshold, tuned_val_metrics = tune_threshold(val_true, val_prob_1)

    print("\n" + "=" * 80)
    print("VALIDATION THRESHOLD TUNING")
    print("=" * 80)
    print(f"Best threshold: {best_threshold:.2f}")
    for key, value in tuned_val_metrics.items():
        print(f"val_tuned_{key}: {value}")

    # Test predictions
    test_output = trainer.predict(tokenized["test"])
    test_logits = test_output.predictions
    test_prob_1 = probs_from_logits(test_logits)
    test_true = test_df["label"].astype(int).values

    # Default threshold 0.50
    test_pred_default = (test_prob_1 >= 0.50).astype(int)
    save_prediction_csv(test_df, test_prob_1, test_pred_default, TEST_PRED_CSV)

    # Tuned threshold
    test_pred_tuned = (test_prob_1 >= best_threshold).astype(int)
    save_prediction_csv(test_df, test_prob_1, test_pred_tuned, TEST_PRED_THRESH_CSV)
    tuned_test_metrics = compute_binary_metrics(test_true, test_pred_tuned)

    print("\n" + "=" * 80)
    print("TEST EVALUATION WITH TUNED THRESHOLD")
    print("=" * 80)
    print(f"Applied threshold: {best_threshold:.2f}")
    for key, value in tuned_test_metrics.items():
        print(f"test_tuned_{key}: {value}")

    # Group-wise best position per id
    save_best_position_per_id(test_df, test_prob_1, BEST_POS_PER_ID_CSV)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("TRAIN RESULT\n")
        for key, value in train_result.metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nSTANDARD VALIDATION METRICS\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nSTANDARD TEST METRICS\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nBEST VALIDATION THRESHOLD\n")
        f.write(f"best_threshold: {best_threshold:.2f}\n")

        f.write("\nTUNED VALIDATION METRICS\n")
        for key, value in tuned_val_metrics.items():
            f.write(f"val_tuned_{key}: {value}\n")

        f.write("\nTUNED TEST METRICS\n")
        for key, value in tuned_test_metrics.items():
            f.write(f"test_tuned_{key}: {value}\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best model saved to : {BEST_MODEL_DIR}")
    print(f"Metrics saved to    : {METRICS_PATH}")
    print(f"Val predictions     : {VAL_PRED_CSV}")
    print(f"Test predictions    : {TEST_PRED_CSV}")
    print(f"Thresholded test    : {TEST_PRED_THRESH_CSV}")
    print(f"Best per id         : {BEST_POS_PER_ID_CSV}")


if __name__ == "__main__":
    main()
