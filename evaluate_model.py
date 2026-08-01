

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("CB_BASE_DIR", PROJECT_DIR))
DATA_DIR = Path(os.getenv("CB_DATA_DIR", PROJECT_DIR))
MODEL_DIR = Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))
BATCH_SIZE  = 32
MAX_LENGTH  = 128
TEST_SIZE   = 0.10    # same untouched test split as training after 80/10/10 split
RANDOM_SEED = 42

LABEL_MAP = {
    "age":                 0,
    "ethnicity":           1,
    "gender":              2,
    "religion":            3,
    "not_cyberbullying":   4,
    "other_cyberbullying": 5,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.keys())

# Possible dataset filenames
DATASET_CANDIDATES = [
    DATA_DIR / "cyberbullying_tweets.csv.xlsx",
    DATA_DIR / "cyberbullying_tweets_csv.xlsx",
    DATA_DIR / "cyberbullying_tweets.csv",
    DATA_DIR / "cyberbullying_tweets.xlsx",
    Path.cwd() / "cyberbullying_tweets.csv.xlsx",
    Path.cwd() / "cyberbullying_tweets_csv.xlsx",
    Path.cwd() / "cyberbullying_tweets.csv",
    Path.cwd() / "cyberbullying_tweets.xlsx",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_dataset():
    for path in DATASET_CANDIDATES:
        if Path(path).is_file():
            logger.info("Dataset found: %s", path)
            return path
    raise FileNotFoundError(
        "Dataset not found. Tried: " + ", ".join(str(p) for p in DATASET_CANDIDATES)
    )


def load_dataset(path):
    ext = Path(path).suffix.lower()
    df  = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["tweet_text", "cyberbullying_type"])
    df["tweet_text"]         = df["tweet_text"].astype(str).str.strip()
    df["cyberbullying_type"] = df["cyberbullying_type"].str.strip().str.lower()
    df = df[df["cyberbullying_type"].isin(LABEL_MAP)]
    return df


def get_test_split(df):
    from sklearn.model_selection import train_test_split
    _, temp_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED,
        stratify=df["cyberbullying_type"]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_df["cyberbullying_type"]
    )
    return test_df


# ── Dataset class ─────────────────────────────────────────────────────────────

def make_ds_class():
    import torch
    from torch.utils.data import Dataset

    class DS(Dataset):
        def __init__(self, texts, labels, tok, max_len):
            self.texts  = texts
            self.labels = labels
            self.tok    = tok
            self.max_len= max_len

        def __len__(self):  return len(self.texts)

        def __getitem__(self, i):
            enc = self.tok(
                self.texts[i], truncation=True,
                max_length=self.max_len, padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label":          torch.tensor(self.labels[i], dtype=torch.long),
            }
    return DS


# ── Run inference ─────────────────────────────────────────────────────────────

def run_inference(test_df):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading tokenizer and model from: %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    labels_int = [LABEL_MAP[l] for l in test_df["cyberbullying_type"]]
    DS         = make_ds_class()
    ds         = DS(test_df["tweet_text"].tolist(), labels_int, tokenizer, MAX_LENGTH)
    dl         = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating", unit="batch"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    return np.array(all_labels), np.array(all_preds)


# ── Compute metrics ───────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    from collections import defaultdict

    n = len(CLASS_NAMES)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        t_name = ID2LABEL[t]
        p_name = ID2LABEL[p]
        if t_name == p_name:
            tp[t_name] += 1
        else:
            fp[p_name] += 1
            fn[t_name] += 1

    per_class = {}
    support   = {}
    for cls in CLASS_NAMES:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        sup  = sum(1 for t in y_true if ID2LABEL[t] == cls)
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1}
        support[cls]   = sup

    accuracy        = float(np.mean(y_true == y_pred))
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall    = float(np.mean([v["recall"]    for v in per_class.values()]))
    macro_f1        = float(np.mean([v["f1"]        for v in per_class.values()]))

    # Weighted F1
    total = sum(support.values())
    weighted_f1 = float(sum(
        per_class[cls]["f1"] * support[cls] / total
        for cls in CLASS_NAMES
    ))

    return {
        "accuracy":         accuracy,
        "macro_precision":  macro_precision,
        "macro_recall":     macro_recall,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_class":        per_class,
        "support":          support,
    }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path=str(BASE_DIR / "confusion_matrix.png")):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping confusion matrix plot")
        logger.warning("Run: pip install matplotlib")
        return

    n      = len(CLASS_NAMES)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    # Normalize for color (row-wise)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_mat = matrix / np.where(row_sums == 0, 1, row_sums)

    short_labels = ["age", "ethnicity", "gender", "religion", "not_cb", "other_cb"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(norm_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Fine-Tuned BERT\nCyberbullying Classification",
                 fontsize=13, fontweight="bold", pad=15)

    thresh = norm_mat.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if norm_mat[i, j] > thresh else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved: %s", save_path)


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(metrics, n_test, report_path="evaluation_report.txt"):
    lines = []
    lines.append("=" * 70)
    lines.append("  EVALUATION REPORT — Cyberbullying Detection System")
    lines.append("  Al-Balqa Applied University | Faculty of AI")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"\n  Model    : {MODEL_DIR}")
    lines.append(f"  Test set : {n_test} samples  ({int(TEST_SIZE*100)}% of dataset)")
    lines.append(f"  Split    : Stratified, seed={RANDOM_SEED}")
    lines.append("\n" + "─" * 70)
    lines.append("  OVERALL METRICS")
    lines.append("─" * 70)
    lines.append(f"  Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    lines.append(f"  Macro Precision   : {metrics['macro_precision']:.4f}")
    lines.append(f"  Macro Recall      : {metrics['macro_recall']:.4f}")
    lines.append(f"  Macro F1-Score    : {metrics['macro_f1']:.4f}")
    lines.append(f"  Weighted F1-Score : {metrics['weighted_f1']:.4f}")
    lines.append("\n" + "─" * 70)
    lines.append("  PER-CLASS METRICS")
    lines.append("─" * 70)
    lines.append(f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>9}")
    lines.append("  " + "─" * 65)
    for cls in CLASS_NAMES:
        m   = metrics["per_class"][cls]
        sup = metrics["support"][cls]
        lines.append(
            f"  {cls:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {sup:>9}"
        )
    lines.append("  " + "─" * 65)
    lines.append(f"  {'macro avg':<22} {metrics['macro_precision']:>10.4f} "
                 f"{metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f} "
                 f"{sum(metrics['support'].values()):>9}")
    lines.append("\n" + "─" * 70)
    lines.append("  INTERPRETATION")
    lines.append("─" * 70)

    f1 = metrics["macro_f1"]
    if f1 >= 0.90:
        grade = "Excellent (ممتاز)"
    elif f1 >= 0.80:
        grade = "Very Good (جيد جداً)"
    elif f1 >= 0.70:
        grade = "Good (جيد)"
    else:
        grade = "Acceptable (مقبول)"

    lines.append(f"  Overall Grade     : {grade}")
    lines.append(f"  Macro F1 = {f1:.4f} — suitable for academic reporting in Chapter 4.")
    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved: %s", report_path)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Model Evaluation — Cyberbullying Classification")
    print("  Al-Balqa Applied University | 2026")
    print("=" * 70 + "\n")

    # 1. Load dataset
    dataset_path = find_dataset()
    df           = load_dataset(dataset_path)
    logger.info("Total dataset: %d rows", len(df))

    # 2. Get test split (same as used in training)
    test_df = get_test_split(df)
    logger.info("Test split: %d rows", len(test_df))

    # 3. Run inference
    y_true, y_pred = run_inference(test_df)

    # 4. Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # 5. Save confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=str(BASE_DIR / "confusion_matrix.png"))

    # 6. Save and print report
    save_report(metrics, n_test=len(test_df), report_path=str(BASE_DIR / "evaluation_report.txt"))

    print("\nFiles saved:")
    print("  evaluation_report.txt  ← for Chapter 4")
    print("  confusion_matrix.png   ← for Figures")