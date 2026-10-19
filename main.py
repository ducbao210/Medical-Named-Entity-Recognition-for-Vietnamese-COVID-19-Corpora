import os
import argparse
import logging
import torch
from accelerate import Accelerator

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TransformerConfig,
    BERTConfig,
    LSTMConfig,
    KDConfig,
    LABEL2ID,
    ID2LABEL,
    LABEL_LIST,
    NUM_LABELS,
    HF_CHECKPOINT_REPO_ID,
)
from src.models import get_tokenizer
from src.pipeline import run_train, run_evaluate, run_infer, run_distill
from src.utils import set_seed, setup_logger
from torch.utils.data import DataLoader
import transformers
from dotenv import load_dotenv
from huggingface_hub import login

transformers.utils.logging.set_verbosity_error()
# Set up main logger
logger = setup_logger("main_orchestrator")


def main():
    # Set seed for reproducibility and load environment variables
    set_seed(seed=42)
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        login(hf_token)
    else:
        logger.warning("HF_TOKEN not found in environment variables.")

    # Initialize configuration classes
    tf_config = TransformerConfig()
    bert_config = BERTConfig()
    lstm_config = LSTMConfig()
    kd_config = KDConfig()

    # Initialize Argument Parser
    parser = argparse.ArgumentParser(
        description="Vietnamese Named Entity Recognition orchestrator pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        nargs="+",
        choices=[
            "phobert",
            "phobert-lora",
            "xlm-roberta",
            "videberta",
            "transformer",
            "lstm",
            "bilstm",
            "all",
        ],
        help="Choose model architecture to work with (use 'all' to train/evaluate all models sequentially).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "infer", "distill"],
        help="Action mode for pipeline execution.",
    )
    parser.add_argument(
        "--use_crf",
        action="store_true",
        default=False,
        help="Attach a Conditional Random Field (CRF) layer to the network output.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override default training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override default training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override default optimizer learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience epochs for early stopping.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load a specific model weights checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=os.getenv("HF_CHECKPOINT_REPO_ID", HF_CHECKPOINT_REPO_ID),
        help="Hugging Face model repo containing checkpoints/ (default: ducbao210/NER).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Optional HF token; defaults to HF_TOKEN from .env.",
    )
    parser.add_argument(
        "--upload_checkpoint",
        action="store_true",
        help="Upload the trained checkpoint(s) to --hf_repo_id after training.",
    )
    parser.add_argument(
        "--infer_text",
        type=str,
        default=None,
        help="Vietnamese text sentence for raw inference.",
    )

    parser.add_argument(
        "--force_cpu_crf",
        action="store_true",
        default=False,
        help="Force CPU usage when using CRF (especially on Apple Silicon) to avoid unsupported operations.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Mixed precision mode; defaults to fp16 on CUDA and no otherwise.",
    )
    # Parse command-line arguments
    args = parser.parse_args()

    if "all" in args.model:
        logger.info("Sequential training and evaluation for all models.")
        base_cfg = None
    else:
        primary_model = args.model[0]
        if "phobert" in primary_model:
            base_cfg = bert_config
        elif primary_model == "transformer":
            base_cfg = tf_config
        else:
            base_cfg = lstm_config

    # Only compute these fallback values when not using the 'all' mode
    batch_size = args.batch_size or (base_cfg.batch_size if base_cfg else None)
    epochs = args.epochs or (base_cfg.epochs if base_cfg else None)
    learning_rate = args.lr or (base_cfg.learning_rate if base_cfg else None)
    max_seq_length = base_cfg.max_seq_length if base_cfg else None
    val_batch_size = base_cfg.val_batch_size if base_cfg else None

    force_cpu = args.use_crf and (
        args.force_cpu_crf
        or (not torch.cuda.is_available() and torch.backends.mps.is_available())
    )
    if force_cpu:
        logger.warning(
            "Detected CRF usage on MPS (Apple Silicon) or the --force_cpu_crf flag is enabled. "
            "Forcing fallback to CPU to prevent asynchronous `torch.logsumexp` errors."
        )
    mixed_precision = args.mixed_precision or (
        "fp16" if torch.cuda.is_available() and not force_cpu else "no"
    )
    if force_cpu and mixed_precision == "fp16":
        parser.error("--mixed_precision fp16 requires GPU, but CRF is forced to CPU")
    accelerator = Accelerator(cpu=force_cpu, mixed_precision=mixed_precision)
    device = accelerator.device

    logger.info(f"Using compute target device: {device}")

    scratch_models = {"lstm", "bilstm", "transformer"}
    needs_phobert_tokenizer = (
        args.mode == "distill"
        or "all" in args.model
        or bool(set(args.model) & scratch_models)
    )
    if needs_phobert_tokenizer:
        logger.info("Loading PhoBERT tokenizer for scratch/KD paths...")
        tokenizer = get_tokenizer("phobert")
    else:
        tokenizer = None  # Each selected pretrained encoder loads its own tokenizer.

    if args.mode == "train":
        run_train(
            args,
            bert_config,
            tf_config,
            lstm_config,
            tokenizer,
            device,
            LABEL2ID,
            accelerator=accelerator,
        )

    elif args.mode == "evaluate":
        run_evaluate(
            args,
            bert_config,
            tf_config,
            lstm_config,
            tokenizer,
            device,
            LABEL2ID,
            ID2LABEL,
            LABEL_LIST,
        )

    elif args.mode == "infer":
        run_infer(
            args, bert_config, tf_config, lstm_config, tokenizer, device, ID2LABEL
        )

    elif args.mode == "distill":
        run_distill(
            args,
            bert_config,
            tf_config,
            lstm_config,
            kd_config,
            tokenizer,
            device,
            LABEL2ID,
            NUM_LABELS,
        )

    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
