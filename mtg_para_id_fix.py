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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
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
OUTPUT_DIR = ROOT / "models/deberta_detection"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
TEST_PRED_CSV = OUTPUT_DIR / "test_predictions.csv"
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
        zero_division=0
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
    print("IMPROVED PAPER-ALIGNED ENTHYMEME DETECTION TRAINING")
    print("=" * 80)
    print(f"Model           : {MODEL_NAME}")
    print(f"Train file      : {TRAIN_CSV}")
    print(f"Validation file : {VAL_CSV}")
    print(f"Test file       : {TEST_CSV}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Best model dir  : {BEST_MODEL_DIR}")
    print(f"Test pred CSV   : {TEST_PRED_CSV}")
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
    print("VALIDATION EVALUATION")
    print("=" * 80)
    val_metrics = trainer.evaluate(tokenized["validation"])
    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    # ---------- Test predictions CSV ----------
    test_output = trainer.predict(tokenized["test"])
    test_logits = test_output.predictions
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_preds = np.argmax(test_logits, axis=-1)

    pred_df = test_df.copy()
    pred_df["true_label"] = pred_df["label"]
    pred_df["pred_label"] = test_preds.astype(int)
    pred_df["prob_0"] = test_probs[:, 0]
    pred_df["prob_1"] = test_probs[:, 1]

    pred_df = pred_df[
        ["id", "input_text", "true_label", "pred_label", "prob_0", "prob_1"]
    ]
    pred_df.to_csv(TEST_PRED_CSV, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("TRAIN RESULT\n")
        for key, value in train_result.metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nVALIDATION METRICS\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTEST METRICS\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best model saved to : {BEST_MODEL_DIR}")
    print(f"Metrics saved to    : {METRICS_PATH}")
    print(f"Test predictions    : {TEST_PRED_CSV}")


if __name__ == "__main__":
    main()
