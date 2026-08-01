

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path(__file__).resolve().parent
BASE_DIR      = Path(os.getenv("CB_BASE_DIR", PROJECT_DIR))
DATA_DIR      = Path(os.getenv("CB_DATA_DIR", PROJECT_DIR))
DATASET_PATH  = Path(os.getenv("CB_DATASET_PATH", DATA_DIR / "cyberbullying_tweets.csv.xlsx"))
OUTPUT_DIR    = Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))
MODEL_NAME    = "bert-base-uncased"
MAX_LENGTH    = 128
BATCH_SIZE    = 16       # قلّل إلى 8 إذا نفدت ذاكرة GPU
NUM_EPOCHS    = 3        # زد إلى 5 للحصول على دقة أعلى
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
SAMPLE_N      = None     # None = كامل الداتاست | رقم = عينة للاختبار السريع

LABEL_MAP = {
    "age":                 0,
    "ethnicity":           1,
    "gender":              2,
    "religion":            3,
    "not_cyberbullying":   4,
    "other_cyberbullying": 5,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


# ── Step 1: Check dependencies ────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "sklearn", "tqdm", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}\n"
            f"Run: pip install {' '.join(missing)}"
        )
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cpu":
        logger.warning(
            "No GPU detected — training on CPU will be SLOW (~2-3 hours). "
            "Consider using Google Colab with GPU runtime."
        )
    return device


# ── Step 2: Load & validate dataset ──────────────────────────────────────────

def load_dataset():
    logger.info("Loading dataset: %s", DATASET_PATH)

    def resolve_dataset_path():
        # 1) exact path as configured
        candidates = [Path(DATASET_PATH)]

        # 2) common naming variants for this dataset
        stem = Path(DATASET_PATH).stem
        candidates.extend(
            [
                DATA_DIR / "cyberbullying_tweets.csv.xlsx",
                DATA_DIR / "cyberbullying_tweets_csv.xlsx",
                DATA_DIR / "cyberbullying_tweets.csv",
                DATA_DIR / "cyberbullying_tweets.xlsx",
                Path.cwd() / "cyberbullying_tweets.csv.xlsx",
                Path.cwd() / "cyberbullying_tweets_csv.xlsx",
                Path.cwd() / "cyberbullying_tweets.csv",
                Path.cwd() / "cyberbullying_tweets.xlsx",
            ]
        )

        # 3) fallback scan for cyberbullying tweets dataset files in data folder and cwd
        candidates.extend(sorted(DATA_DIR.glob("cyberbullying_tweets*.*")))
        candidates.extend(sorted(Path.cwd().glob("cyberbullying_tweets*.*")))

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    dataset_file = resolve_dataset_path()
    if dataset_file is None:
        raise FileNotFoundError(
            "Dataset not found.\n"
            f"Tried default: {DATASET_PATH}\n"
            f"Working directory: {os.getcwd()}\n"
            "Expected one of: cyberbullying_tweets*.xlsx/.xls/.csv"
        )

    logger.info("Using dataset file: %s", dataset_file)
    ext = dataset_file.suffix.lower()
    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(dataset_file)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel datasets requires 'openpyxl'. "
                "Install it with: pip install openpyxl"
            ) from exc
    else:
        df = pd.read_csv(dataset_file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate columns
    for col in ["tweet_text", "cyberbullying_type"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

    df = df.dropna(subset=["tweet_text", "cyberbullying_type"])
    df["tweet_text"]       = df["tweet_text"].astype(str).str.strip()
    df["cyberbullying_type"] = df["cyberbullying_type"].str.strip().str.lower()

    # Remove unknown labels
    valid_labels = set(LABEL_MAP.keys())
    unknown = set(df["cyberbullying_type"].unique()) - valid_labels
    if unknown:
        logger.warning("Removing rows with unknown labels: %s", unknown)
        df = df[df["cyberbullying_type"].isin(valid_labels)]

    if SAMPLE_N:
        df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)

    logger.info("Dataset: %d rows", len(df))
    logger.info("Label distribution:\n%s", df["cyberbullying_type"].value_counts().to_string())

    return df


# ── Step 3: Dataset class ─────────────────────────────────────────────────────

def make_dataset_class():
    from torch.utils.data import Dataset
    import torch

    class CyberbullyingDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts    = texts
            self.labels   = labels
            self.tok      = tokenizer
            self.max_len  = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return CyberbullyingDataset


# ── Step 4: Train ─────────────────────────────────────────────────────────────

def train(df, device):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from tqdm import tqdm
    import torch

    CyberbullyingDataset = make_dataset_class()

    # ── Split ────────────────────────────────────────────────────────────────
    # Use 80/10/10 stratified split. The validation set is used for checkpointing;
    # the test set is kept untouched for the final report to avoid data leakage.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["cyberbullying_type"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["cyberbullying_type"],
    )
    logger.info("Train: %d | Validation: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Prepare data ─────────────────────────────────────────────────────────
    def make_labels(df_):
        return [LABEL_MAP[lbl] for lbl in df_["cyberbullying_type"]]

    train_ds = CyberbullyingDataset(
        train_df["tweet_text"].tolist(), make_labels(train_df), tokenizer, MAX_LENGTH
    )
    val_ds = CyberbullyingDataset(
        val_df["tweet_text"].tolist(), make_labels(val_df), tokenizer, MAX_LENGTH
    )
    test_ds = CyberbullyingDataset(
        test_df["tweet_text"].tolist(), make_labels(test_df), tokenizer, MAX_LENGTH
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )
    model.to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dl) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training — %d epochs, %d steps total", NUM_EPOCHS, total_steps)
    best_f1    = 0.0
    train_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        correct     = 0
        total       = 0
        t_start     = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds       = outputs.logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{correct/total:.3f}",
            })

        avg_loss     = total_loss / len(train_dl)
        train_acc    = correct / total
        elapsed      = time.time() - t_start

        logger.info(
            "Epoch %d/%d — Loss: %.4f | Train Acc: %.4f | Time: %.1fs",
            epoch, NUM_EPOCHS, avg_loss, train_acc, elapsed
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate_model(model, val_dl, device)
        logger.info(
            "Validation — Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            val_metrics["accuracy"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
        )

        train_history.append({
            "epoch":     epoch,
            "loss":      round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            **val_metrics,
        })

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model(model, tokenizer)
            logger.info("✓ Best model saved (F1=%.4f)", best_f1)

    test_metrics = evaluate_model(model, test_dl, device)
    logger.info(
        "Final Test — Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        test_metrics["accuracy"], test_metrics["f1"],
        test_metrics["precision"], test_metrics["recall"],
    )
    return train_history, test_df, tokenizer, test_metrics


# ── Step 5: Evaluate ─────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device):
    import torch
    from collections import defaultdict

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds  = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Convert to label names
    y_pred = [ID2LABEL[p] for p in all_preds]
    y_true = [ID2LABEL[l] for l in all_labels]

    # Compute metrics
    classes = sorted(set(y_true) | set(y_pred))
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p: tp[t] += 1
        else:      fp[p] += 1; fn[t] += 1

    precs, recs, f1s = [], [], []
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(np.mean(precs), 4),
        "recall":    round(np.mean(recs),  4),
        "f1":        round(np.mean(f1s),   4),
    }


# ── Step 6: Save model ────────────────────────────────────────────────────────

def save_model(model, tokenizer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Verify config has correct labels
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["id2label"] = {str(k): v for k, v in ID2LABEL.items()}
    config["label2id"] = LABEL_MAP

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Model saved to: %s", OUTPUT_DIR)


# ── Step 7: Final report ──────────────────────────────────────────────────────

def print_report(history, test_metrics=None):
    print("\n" + "═" * 60)
    print("  TRAINING REPORT")
    print("═" * 60)
    print(f"  {'Epoch':<8} {'Loss':<10} {'Train Acc':<12} {'Val Acc':<10} {'Val F1'}")
    print("  " + "─" * 55)
    for h in history:
        print(
            f"  {h['epoch']:<8} {h['loss']:<10.4f} "
            f"{h['train_acc']:<12.4f} {h['accuracy']:<10.4f} {h['f1']:.4f}"
        )
    best = max(history, key=lambda x: x["f1"])
    print("═" * 60)
    print(f"  Best F1: {best['f1']:.4f} at Epoch {best['epoch']}")
    print(f"  Model saved to: {OUTPUT_DIR}")
    if test_metrics is not None:
        print("─" * 60)
        print(
            f"  Final Test | Acc: {test_metrics['accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f}"
        )
    print("═" * 60)
    print()
    print("  To use the trained model:")
    print("  ─────────────────────────")
    print('  from cyberbullying_detection_system import (')
    print('      IntegratedCyberbullyingDetectionSystem, BERTToxicityAnalyzer')
    print('  )')
    print(f'  bert = BERTToxicityAnalyzer(model_name="{OUTPUT_DIR}")')
    print('  system = IntegratedCyberbullyingDetectionSystem(bert_analyzer=bert)')
    print('  result = system.analyze("Your text here")')
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BERT Fine-Tuning — Cyberbullying Classification")
    print("  Al-Balqa Applied University | 2026")
    print("=" * 60 + "\n")

    # 1. Check environment
    device = check_deps()

    # 2. Load data
    df = load_dataset()

    # 3. Train
    history, test_df, tokenizer, test_metrics = train(df, device)

    # 4. Final report
    print_report(history, test_metrics)

    print("Training complete! ✓")
    print(f"Run test_model.py to evaluate the saved model.")
