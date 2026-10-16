import os
import csv
import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    classification_report as sk_classification_report,
    accuracy_score as sk_accuracy_score,
    confusion_matrix,
)

from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch; enable deterministic CUDA operations."""
    # Must be set before the first CUDA context is initialized.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --- NER EVALUATION METRICS ---


def compute_entity_level_metrics(preds, labels, id2label):
    """Compute span/entity-level precision, recall, and F1 using seqeval. (expects a 2D list)"""
    true_predictions = [
        [id2label[p] for (p, l) in zip(pr, lb) if l != -100]
        for pr, lb in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pr, lb) if l != -100]
        for pr, lb in zip(preds, labels)
    ]

    # Provide overall precision, recall, and f1-score under 'overall' key for compatibility
    return {
        "overall": {
            "precision": precision_score(
                true_labels, true_predictions, mode="strict", scheme=IOB2
            ),
            "recall": recall_score(
                true_labels, true_predictions, mode="strict", scheme=IOB2
            ),
            "f1-score": f1_score(
                true_labels, true_predictions, mode="strict", scheme=IOB2
            ),
        },
        "report": seqeval_classification_report(
            true_labels, true_predictions, mode="strict", scheme=IOB2
        ),
    }


def compute_metrics(preds, labels, label_list):
    """Compute token-level metrics using sklearn. (expects flattened 1D arrays)"""
    # Filter out padding/subword positions marked with -100
    f_preds = [p for p, l in zip(preds, labels) if l != -100]
    f_labels = [l for p, l in zip(preds, labels) if l != -100]

    id2label = {i: l for i, l in enumerate(label_list)}

    # Convert IDs to string labels
    if any(p not in id2label for p in f_preds) or any(
        l not in id2label for l in f_labels
    ):
        raise ValueError("Metric input contains an invalid label ID")
    f_preds_str = [id2label[p] for p in f_preds]
    f_labels_str = [id2label[l] for l in f_labels]

    # Fixed label set/order makes metrics comparable across runs.
    present_labels = list(label_list)

    # Use sklearn utilities to compute classification metrics
    report = sk_classification_report(
        f_labels_str,
        f_preds_str,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = sk_accuracy_score(f_labels_str, f_preds_str)

    return report


# --- DATA PREPARATION & MODEL DIAGNOSTICS ---


def calculate_label_weights(train_file, label2id):
    """Compute class weights from a training file using inverse frequency

    to handle class imbalance.
    """
    counts = {l: 0 for l in label2id.keys()}
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                tag = parts[-1]
                if tag in counts:
                    counts[tag] += 1

    total = sum(counts.values())
    weights = []
    for label in sorted(label2id, key=label2id.get):
        count = counts[label]
        w = total / (count + 1e-5)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float)
    return weights / weights.mean()


def print_model_size(model, model_name="Model"):
    """Print a structured summary of model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f" {model_name} Parameters")
    print(f"{'='*50}")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {total - trainable:>12,}")
    print(f"  Trainable ratio:      {trainable/total*100:>11.2f}%")
    print(f"{'='*50}\n")
    return total, trainable


# --- VISUALIZATION UTILITIES ---


def plot_loss_curve(history, save_path="results/plots/loss_curve.png"):
    """Plot training and validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", marker="s", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved loss plot to {save_path}")


def plot_acc_curve(history, save_path="results/plots/acc_curve.png"):
    """Plot training and validation accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy", marker="o", linewidth=2)
    plt.plot(history["val_acc"], label="Validation Accuracy", marker="s", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training & Validation Accuracy Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved accuracy plot to {save_path}")


def plot_f1_curve(history, save_path="results/plots/f1_curve.png"):
    """Plot F1-score evolution over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_f1"], label="Validation F1", marker="s", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    plt.title("F1 Score Evolution", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved F1 plot to {save_path}")


def plot_confusion_matrix(
    preds, labels, label_list, save_path="results/plots/cm.png", ignore_index=-100
):
    """Plot a confusion matrix restricted to entity labels (excludes 'O')."""
    id2label = {i: l for i, l in enumerate(label_list)}

    # Auto-convert 1D (flat) arrays to 2D if necessary
    if len(preds) > 0 and not isinstance(
        preds[0], (list, tuple, np.ndarray, torch.Tensor)
    ):
        preds = [preds]
        labels = [labels]

    filtered = [
        (p, l)
        for pr, lb in zip(preds, labels)
        for p, l in zip(pr, lb)
        if l != ignore_index
    ]
    if not filtered:
        return
    f_preds, f_labels = zip(*filtered)

    entity_labels = [l for l in label_list if l != "O"]
    entity_ids = [label_list.index(l) for l in entity_labels]

    mask = [(l in entity_ids or p in entity_ids) for p, l in zip(f_preds, f_labels)]
    if not any(mask):
        return

    entity_preds = [id2label.get(p, "O") for p, m in zip(f_preds, mask) if m]
    entity_labels_filtered = [id2label.get(l, "O") for l, m in zip(f_labels, mask) if m]
    present_labels = sorted(set(entity_preds) | set(entity_labels_filtered))

    cm = confusion_matrix(entity_labels_filtered, entity_preds, labels=present_labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix (Entity Labels Excl. 'O')", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved confusion matrix to {save_path}")


def plot_entity_distribution(
    labels, label_list, save_path="results/plots/distribution.png", ignore_index=-100
):
    """Plot the distribution of entity labels (excluding 'O')."""
    id2label = {i: l for i, l in enumerate(label_list)}

    # Auto-convert 1D (flat) arrays to 2D if necessary
    if len(labels) > 0 and not isinstance(
        labels[0], (list, tuple, np.ndarray, torch.Tensor)
    ):
        labels = [labels]

    filtered = [
        id2label.get(l, "O") for seq in labels for l in seq if l != ignore_index
    ]

    entity_labels = [l for l in filtered if l != "O"]

    if not entity_labels:
        return

    counter = Counter(entity_labels)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels_sorted, counts = zip(*sorted_items)

    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_sorted)))
    plt.bar(labels_sorted, counts, color=colors)
    plt.xlabel("Entity Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Entity Label Distribution", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved entity distribution to {save_path}")


# --- FILE EXPORTS & ANALYSIS ---


def save_metrics_csv(metrics_dict, save_path="results/metrics.csv"):
    """Save training metrics dictionary to a CSV file using pandas."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Remove the 'report' key (complex nested string) before saving to CSV
    clean_dict = {k: v for k, v in metrics_dict.items() if k != "report"}
    df = pd.DataFrame([clean_dict])
    df.to_csv(save_path, index=False)
    print(f"  -> Saved metrics to {save_path}")


def save_predictions(sentences, true_labels, pred_labels, save_path, id2label):
    """Save token-level predictions to a TSV text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for sent, trues, preds in zip(sentences, true_labels, pred_labels):
            for token, t, p in zip(sent, trues, preds):
                t_name = id2label.get(t, "O") if isinstance(t, int) else t
                p_name = id2label.get(p, "O") if isinstance(p, int) else p
                f.write(f"{token}\t{t_name}\t{p_name}\n")
            f.write("\n")
    print(f"  -> Saved predictions to {save_path}")


def get_error_analysis(sentences, true_labels, pred_labels, id2label, num_samples=5):
    """Print error cases for human inspection and error profiling."""
    print(f"\n{'='*50}\n ERROR ANALYSIS\n{'='*50}")
    errors_found = 0

    for sent, trues, preds in zip(sentences, true_labels, pred_labels):
        if errors_found >= num_samples:
            break

        has_error = False
        sent_errors = []
        for i, (t, p) in enumerate(zip(trues, preds)):
            if t != p and t != -100:
                has_error = True
                sent_errors.append(
                    {
                        "word": sent[i] if i < len(sent) else "<PAD>",
                        "true": id2label.get(t, "O"),
                        "pred": id2label.get(p, "O"),
                    }
                )

        if has_error:
            print(f"Sentence: {' '.join(sent)}")
            for detail in sent_errors:
                print(
                    f"  -> Word: '{detail['word']}' | True: {detail['true']} | Pred: {detail['pred']}"
                )
            print("-" * 30)
            errors_found += 1


def setup_logger(name, log_dir="results/logs", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance with both console and file handlers.

    This function ensures that a log directory exists and generates a log file
    named with the current date if no specific file is provided.

    Parameters
    ---
    name : str
            The name of the logger (usually __name__).
    log_dir : str, optional
            Directory to save log files. Defaults to "logs".
    log_file : str, optional
            Specific path for the log file. If None, a default name based on the date is generated.
    level : int, optional
            The logging level (e.g., logging.INFO). Defaults to logging.INFO.

    Returns
    ---
    logging.Logger
            A configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Use the supplied `log_dir`, or fall back to the package default when empty.
    if log_dir is None or str(log_dir).strip() == "":
        log_dir_path = Path("results/logs")
    else:
        log_dir_path = Path(log_dir)

    # File handler
    if log_file is None:
        os.makedirs(log_dir_path, exist_ok=True)

        # File name: modelname_YYYY-MM-DD_HH-MM-SS.log
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir_path, f"{name}_{current_time}.log")
    else:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except (TypeError, OSError):
            pass

    file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
