from pathlib import Path
from dataclasses import dataclass
import os

# --- Base directory ---
BASE_DIR = Path(__file__).resolve().parents[2]

# --- Data paths ---
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train_word.conll"
DEV_FILE = DATA_DIR / "dev_word.conll"
TEST_FILE = DATA_DIR / "test_word.conll"

# --- Hugging Face checkpoint defaults ---
# Override with HF_CHECKPOINT_REPO_ID in .env when using another repository.
HF_CHECKPOINT_REPO_ID = os.getenv("HF_CHECKPOINT_REPO_ID", "ducbao210/NER")

# --- Output paths ---
# All models share these folders. The model/CRF identity is encoded in filenames.
OUTPUT_DIR = BASE_DIR / "results"
RESULT_DIRS = {
    "checkpoints": OUTPUT_DIR / "checkpoints",
    "logs": OUTPUT_DIR / "logs",
    "plots": OUTPUT_DIR / "plots",
    "tensorboard": OUTPUT_DIR / "tensorboard",
    "predictions": OUTPUT_DIR / "predictions",
    "metrics": OUTPUT_DIR / "metrics",
}


def model_prefix(model_name: str, use_crf: bool = False) -> str:
    """Return the stable filename prefix used by every model artifact."""
    return f"{model_name}{'_crf' if use_crf else ''}"


def get_model_dirs(model_name: str, use_crf: bool = False):
    """Return shared output directories and model-prefixed artifact paths."""
    prefix = model_prefix(model_name, use_crf)
    for path in RESULT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base": str(OUTPUT_DIR),
        "prefix": prefix,
        **{name: str(path) for name, path in RESULT_DIRS.items()},
        "checkpoint_best": str(RESULT_DIRS["checkpoints"] / f"{prefix}_best_model.pt"),
        "checkpoint_latest": str(
            RESULT_DIRS["checkpoints"] / f"{prefix}_latest_model.pt"
        ),
    }


# --- Label Definitions ---
# Exactly matching unique tags found in the PhoNER_COVID19 dataset
LABEL_LIST = [
    "O",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-SYMPTOM_AND_DISEASE",
    "I-SYMPTOM_AND_DISEASE",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-PATIENT_ID",
    "I-PATIENT_ID",
    "B-AGE",
    "I-AGE",
    "B-NAME",
    "I-NAME",
    "B-JOB",
    "I-JOB",
    "B-TRANSPORTATION",
    "I-TRANSPORTATION",
    "B-GENDER",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# --- Native Transformer Hyperparameters ---
@dataclass
class TransformerConfig:
    embedding_dim: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_length: int = 256
    batch_size: int = 32
    val_batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 5e-4
    patience: int = 5


# --- BERT & PhoBERT Hyperparameters ---
@dataclass
class BERTConfig:
    max_seq_length: int = 256
    batch_size: int = 16
    val_batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    patience: int = 3
    weight_decay: float = 0.01


# --- LSTM & Bi-LSTM Hyperparameters ---
@dataclass
class LSTMConfig:
    embedding_dim: int = 300
    hidden_dim: int = 256
    dropout: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 20
    max_seq_length: int = 256
    batch_size: int = 32
    val_batch_size: int = 32
    patience: int = 5


# --- Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configurations ---
@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    warmup_ratio: float = 0.1


# --- Knowledge Distillation (KD) Configurations ---
@dataclass
class KDConfig:
    temperature: float = 4.0
    alpha: float = 0.7
