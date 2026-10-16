"""Checkpoint naming, validation, upload and lazy download helpers."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, upload_file

from src.config import HF_CHECKPOINT_REPO_ID, RESULT_DIRS, model_prefix


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be resolved or is invalid."""


def checkpoint_filenames(model_name: str, use_crf: bool = False) -> tuple[str, str]:
    prefix = model_prefix(model_name, use_crf)
    return f"{prefix}_best_model.pt", f"{prefix}_latest_model.pt"


def validate_hf_token(token: Optional[str] = None, require_write: bool = False) -> dict:
    """Validate authentication without exposing the token.

    Read access is checked with ``whoami``. Write access is checked by creating
    or resolving the target repository in ``upload_checkpoints``; HF does not
    expose a reliable portable scope field for every token type.
    """
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise CheckpointError("HF_TOKEN is required for Hugging Face checkpoint access.")
    try:
        return HfApi(token=token).whoami()
    except Exception as exc:
        mode = "read/write" if require_write else "read"
        raise CheckpointError(f"Hugging Face {mode} token validation failed: {exc}") from exc


def _checkpoint_dir() -> Path:
    path = Path(RESULT_DIRS["checkpoints"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_local_checkpoint(model_name: str, use_crf: bool, requested: Optional[str] = None) -> Optional[Path]:
    """Resolve an explicit path or the model-prefixed best/latest local file."""
    if requested:
        candidate = Path(requested)
        if candidate.is_file():
            return candidate
        return None
    directory = _checkpoint_dir()
    for filename in checkpoint_filenames(model_name, use_crf):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def download_checkpoint(
    model_name: str,
    use_crf: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Download one model checkpoint into the same local path used by training."""
    repo_id = repo_id or os.getenv("HF_CHECKPOINT_REPO_ID") or HF_CHECKPOINT_REPO_ID
    if not repo_id:
        raise CheckpointError("Set HF_CHECKPOINT_REPO_ID or pass repo_id to download checkpoints.")
    validate_hf_token(token)
    best_name, latest_name = checkpoint_filenames(model_name, use_crf)
    target_name = filename or best_name
    if target_name not in {best_name, latest_name}:
        raise CheckpointError(f"Unsupported checkpoint filename: {target_name}")

    # Default inference prefers best; if the remote repo has only latest, use it.
    candidates = [target_name]
    if filename is None:
        candidates.append(latest_name if target_name == best_name else best_name)
    errors = []
    for candidate_name in candidates:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=f"checkpoints/{candidate_name}",
                repo_type="model",
                token=token or os.getenv("HF_TOKEN"),
                revision=revision,
            )
            destination = _checkpoint_dir() / candidate_name
            shutil.copy2(downloaded, destination)
            return destination
        except Exception as exc:
            errors.append(f"{candidate_name}: {exc}")
    raise CheckpointError(
        f"Could not download a checkpoint for {model_name} from {repo_id}. "
        f"Expected checkpoints/{best_name} and/or checkpoints/{latest_name}. "
        + " | ".join(errors)
    )


def ensure_checkpoint(
    model_name: str,
    use_crf: bool = False,
    requested: Optional[str] = None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """Use local checkpoint first; lazily download it when local storage is empty."""
    local = resolve_local_checkpoint(model_name, use_crf, requested)
    if local:
        return local
    if requested and not (repo_id or os.getenv("HF_CHECKPOINT_REPO_ID") or HF_CHECKPOINT_REPO_ID):
        raise CheckpointError(f"Checkpoint does not exist: {requested}")
    requested_name = Path(requested).name if requested else None
    allowed = set(checkpoint_filenames(model_name, use_crf))
    filename = requested_name if requested_name in allowed else None
    return download_checkpoint(
        model_name, use_crf, repo_id=repo_id, token=token, filename=filename
    )


def upload_checkpoints(
    repo_id: str,
    model_name: Optional[str] = None,
    use_crf: bool = False,
    token: Optional[str] = None,
    private: Optional[bool] = None,
    commit_message: str = "Upload NER checkpoints",
) -> list[str]:
    """Upload prefixed checkpoints under ``checkpoints/`` in a HF model repo."""
    token = token or os.getenv("HF_TOKEN")
    validate_hf_token(token, require_write=True)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    directory = _checkpoint_dir()
    names = checkpoint_filenames(model_name, use_crf) if model_name else tuple(
        sorted(p.name for p in directory.glob("*.pt"))
    )
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise CheckpointError(
            "Upload requires both best and latest checkpoints; missing: " + ", ".join(missing)
        )
    uploaded: list[str] = []
    for name in names:
        path = directory / name
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"checkpoints/{name}",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
        uploaded.append(f"checkpoints/{name}")
    if not uploaded:
        raise CheckpointError(f"No .pt checkpoint found in {directory}")
    manifest = {
        "format": "torch_state_dict_v1",
        "files": uploaded,
        "source": "Medical-Named-Entity-Recognition-for-Vietnamese-COVID-19-Corpora",
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="checkpoints/manifest.json",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )
    return uploaded
