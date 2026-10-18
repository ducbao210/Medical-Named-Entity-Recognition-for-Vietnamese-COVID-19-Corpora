import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from src.config import (
    LoRAConfig,
    NUM_LABELS,
    LABEL2ID,
    LSTMConfig,
    TransformerConfig,
    BERTConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.utils import setup_logger
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = setup_logger("model")

PRETRAINED_MODEL_IDS = {
    "phobert": "vinai/phobert-base",
    "phobert-lora": "vinai/phobert-base",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "videberta": "Fsoft-AIC/videberta-base",
}


def get_tokenizer(model_name, fallback=None):
    """Load a tokenizer matched to the selected pretrained encoder."""
    checkpoint = PRETRAINED_MODEL_IDS.get(model_name)
    if fallback is not None and model_name in ("phobert", "phobert-lora"):
        return fallback
    if checkpoint is None:
        if fallback is None:
            raise ValueError(f"No tokenizer configured for model {model_name!r}")
        return fallback
    if model_name in ("phobert", "phobert-lora"):
        # PhoBERT has no official fast tokenizer; use its canonical BPE vocab.
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_fast=False, keep_accents=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if not getattr(tokenizer, "is_fast", False) and model_name not in (
        "phobert",
        "phobert-lora",
    ):
        raise RuntimeError(
            f"Model {model_name!r} requires a fast tokenizer for word_ids alignment"
        )
    return tokenizer


def configure_encoder_trainability(encoder, trainable_layers=2):
    """Freeze the encoder, then optionally unfreeze only its final N blocks."""
    if trainable_layers is not None and trainable_layers < 0:
        raise ValueError("trainable_layers must be non-negative or None")
    for param in encoder.parameters():
        param.requires_grad = False

    if trainable_layers is None:
        for param in encoder.parameters():
            param.requires_grad = True
        return None
    if trainable_layers == 0:
        return []

    layer_stack = None
    for parent_name in ("encoder", "transformer"):
        parent = getattr(encoder, parent_name, None)
        if parent is None:
            continue
        for stack_name in ("layer", "layers"):
            candidate = getattr(parent, stack_name, None)
            if isinstance(candidate, (nn.ModuleList, nn.Sequential)) and len(candidate):
                layer_stack = candidate
                break
        if layer_stack is not None:
            break

    if layer_stack is None:
        raise RuntimeError(
            f"Cannot locate transformer block stack for {encoder.__class__.__name__}; "
            "refusing to silently train/freeze the wrong parameters"
        )

    start = max(0, len(layer_stack) - trainable_layers)
    selected = list(range(start, len(layer_stack)))
    for layer in layer_stack[start:]:
        for param in layer.parameters():
            param.requires_grad = True
    return selected


class NERLoss(nn.Module):
    """
    Unified loss function that routes labels and logits depending on whether
    the model utilizes a CRF decoding layer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, labels, inputs=None):
        if hasattr(self.model, "use_crf") and self.model.use_crf:
            return self.model.crf_loss(outputs, labels, inputs)
        else:
            return self.ce(outputs.view(-1, outputs.shape[-1]), labels.view(-1))


def get_model(model_name, vocab_size, pad_token_id=1, use_crf=False):
    """Instantiate the requested model class."""
    if model_name == "lstm":
        lstm_cfg = LSTMConfig()
        logger.info(f"Instantiating LSTM model (CRF={use_crf})")
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    elif model_name == "bilstm":
        logger.info(f"Instantiating Bidirectional LSTM model (CRF={use_crf})")
        lstm_cfg = LSTMConfig()
        return BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    elif model_name == "phobert":
        logger.info(f"Instantiating PhoBERT model (CRF={use_crf})")
        return PhoBERTModel(
            num_labels=NUM_LABELS,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
            trainable_layers=2,
        )
    elif model_name in ("xlm-roberta", "videberta"):
        logger.info(f"Instantiating {model_name} (CRF={use_crf})")
        return PretrainedTokenClassifier(
            model_name=PRETRAINED_MODEL_IDS[model_name],
            num_labels=NUM_LABELS,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
            trainable_layers=2,
        )
    elif model_name == "phobert-lora":
        logger.info(f"Instantiating PhoBERT + LoRA model (CRF={use_crf})")
        return PhoBERTLoRA(
            num_labels=NUM_LABELS, pad_token_id=pad_token_id, use_crf=use_crf
        )
    elif model_name == "transformer":
        logger.info(f"Instantiating Scratch Transformer model (CRF={use_crf})")
        tf_cfg = TransformerConfig()
        return TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=tf_cfg.embedding_dim,
            nhead=tf_cfg.nhead,
            num_layers=tf_cfg.num_layers,
            dim_feedforward=tf_cfg.dim_feedforward,
            num_labels=NUM_LABELS,
            max_seq_length=tf_cfg.max_seq_length,
            dropout=tf_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class CRFLayer(nn.Module):
    """Linear-chain CRF that supports arbitrary (including gapped) masks."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.01, 0.01)
        nn.init.uniform_(self.start_transitions, -0.01, 0.01)
        nn.init.uniform_(self.end_transitions, -0.01, 0.01)

    def _active(self, emissions, tags, mask):
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )
        mask = mask.bool()
        if mask.shape != emissions.shape[:2]:
            raise ValueError("CRF mask must have shape (batch, sequence_length)")
        sequences = []
        for b in range(emissions.size(0)):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                raise ValueError(f"CRF sequence {b} has no active positions")
            sequences.append(
                (
                    emissions[b].index_select(0, idx),
                    tags[b].index_select(0, idx) if tags is not None else None,
                )
            )
        return sequences

    def forward(self, emissions, tags, mask=None):
        active_sequences = self._active(emissions, tags, mask)
        losses = []
        total_active = 0
        for seq, gold in active_sequences:
            total_active += seq.size(0)
            alpha = self.start_transitions + seq[0]
            gold_score = self.start_transitions[gold[0]] + seq[0, gold[0]]
            for t in range(1, len(gold)):
                alpha = torch.logsumexp(
                    alpha[:, None] + self.transitions.transpose(0, 1) + seq[t][None, :],
                    dim=0,
                )
                gold_score = (
                    gold_score
                    + self.transitions[gold[t], gold[t - 1]]
                    + seq[t, gold[t]]
                )
            log_z = torch.logsumexp(alpha + self.end_transitions, dim=0)
            losses.append(log_z - gold_score)
        return torch.stack(losses).sum() / max(total_active, 1)

    def decode(self, emissions, mask=None):
        paths = []
        for seq, _ in self._active(emissions, None, mask):
            score = self.start_transitions + seq[0]
            back = []
            for t in range(1, seq.size(0)):
                candidates = score[:, None] + self.transitions.transpose(0, 1)
                best_score, best_from = candidates.max(dim=0)
                score = best_score + seq[t]
                back.append(best_from)
            tag = (score + self.end_transitions).argmax().item()
            path = [tag]
            for ptr in reversed(back):
                tag = ptr[tag].item()
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths


class LSTMModel(nn.Module):
    """
    A standard LSTM model with an optional CRF classification layer for NER baseline.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_labels,
        dropout=0.5,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.pad_token_id
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.masked_fill(~mask, 0)
        # Compute CRF dynamic programming in FP32 for numerical stability.
        return self.crf(logits.float(), clean_labels, mask)

    def decode(self, input_ids, mask=None):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                active_mask = (
                    mask if mask is not None else (input_ids != self.pad_token_id)
                )
                return self.crf.decode(logits, active_mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()


class BiLSTMModel(nn.Module):
    """
    A Bidirectional LSTM model with an optional CRF classification layer for NER.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_labels,
        dropout=0.5,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.pad_token_id
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        lengths = (input_ids != self.pad_token_id).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.masked_fill(~mask, 0)
        # Compute CRF dynamic programming in FP32 for numerical stability.
        return self.crf(logits.float(), clean_labels, mask)

    def decode(self, input_ids, mask=None):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                active_mask = (
                    mask if mask is not None else (input_ids != self.pad_token_id)
                )
                return self.crf.decode(logits, active_mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()


class PhoBERTModel(nn.Module):
    """
    Pretrained transformer token tagger with an optional CRF classification layer.
    """

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=NUM_LABELS,
        pad_token_id=1,
        use_crf=False,
        trainable_layers=2,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        # Keep master weights in FP32; autocast handles lower-precision forward
        # operations while GradScaler requires FP32 trainable gradients.
        self.bert = AutoModel.from_pretrained(model_name).float()
        selected_layers = configure_encoder_trainability(self.bert, trainable_layers)
        self.trainable_encoder_layers = selected_layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

        total = sum(param.numel() for param in self.bert.parameters())
        trainable = sum(
            param.numel() for param in self.bert.parameters() if param.requires_grad
        )
        logger.info(
            "%s encoder: %s/%s parameters trainable; top blocks=%s",
            model_name,
            f"{trainable:,}",
            f"{total:,}",
            selected_layers,
        )

    def forward(self, input_ids):
        # Build the attention mask from this tokenizer's configured padding ID.
        attention_mask = (input_ids != self.pad_token_id).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.masked_fill(~mask, 0)
        # Compute CRF dynamic programming in FP32 for numerical stability.
        return self.crf(logits.float(), clean_labels, mask)

    def decode(self, input_ids, mask=None):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                active_mask = (
                    mask if mask is not None else (input_ids != self.pad_token_id)
                )
                return self.crf.decode(logits, active_mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()


class PretrainedTokenClassifier(PhoBERTModel):
    """Token-classification head over any AutoModel-compatible encoder."""


class PhoBERTLoRA(nn.Module):
    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=NUM_LABELS,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.base_model = PhoBERTModel(
            model_name=model_name,
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
            trainable_layers=0,
        )

        # `phobert-lora`: keep every original pretrained encoder weight frozen.
        for param in self.base_model.bert.parameters():
            param.requires_grad = False
        cfg = LoRAConfig()
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=["query", "value"],
        )

        self.base_model.bert = get_peft_model(self.base_model.bert, peft_config)
        for name, param in self.base_model.bert.named_parameters():
            param.requires_grad = "lora_" in name
        if not any(param.requires_grad for param in self.base_model.bert.parameters()):
            raise RuntimeError("LoRA adapter parameters were not made trainable")
        if any(
            param.requires_grad and "lora_" not in name
            for name, param in self.base_model.bert.named_parameters()
        ):
            raise RuntimeError(
                "A pretrained base encoder parameter was unexpectedly unfrozen"
            )

        # Unfreeze classifier and optional CRF task head.
        logger.info(
            "PhoBERT-LoRA: original encoder frozen; adapters target query/value only"
        )

        # Unfreeze classifier and crf
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

        if use_crf and hasattr(self.base_model, "crf"):
            for param in self.base_model.crf.parameters():
                param.requires_grad = True

    @property
    def use_crf(self):
        return self.base_model.use_crf

    def forward(self, input_ids):
        return self.base_model(input_ids)

    def crf_loss(self, logits, labels, input_ids=None):
        return self.base_model.crf_loss(logits, labels, input_ids)

    def decode(self, input_ids, mask=None):
        return self.base_model.decode(input_ids, mask=mask)


class PositionalEncoding(nn.Module):
    """Positional encoding layer for the from-scratch Transformer."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Pure Transformer (encoder-only) model with optional CRF for NER tasks."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        num_layers,
        dim_feedforward,
        num_labels,
        max_seq_length=256,
        dropout=0.1,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_length, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.classifier = nn.Linear(embedding_dim, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # Create key padding mask from input_ids (1 is PhoBERT's pad token)
        src_key_padding_mask = input_ids == self.pad_token_id

        embeds = self.embedding(input_ids)
        embeds = self.pos_encoder(embeds)

        encoder_out = self.transformer_encoder(
            embeds, src_key_padding_mask=src_key_padding_mask
        )

        logits = self.classifier(encoder_out)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.masked_fill(~mask, 0)
        # Compute CRF dynamic programming in FP32 for numerical stability.
        return self.crf(logits.float(), clean_labels, mask)

    def decode(self, input_ids, mask=None):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                active_mask = (
                    mask if mask is not None else (input_ids != self.pad_token_id)
                )
                return self.crf.decode(logits, active_mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()
