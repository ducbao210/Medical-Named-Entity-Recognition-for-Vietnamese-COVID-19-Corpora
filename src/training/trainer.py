import os
import shutil
from contextlib import nullcontext
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.utils import setup_logger, print_model_size, compute_entity_level_metrics
from src.config import ID2LABEL


def _safe_torch_load(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


class BaseTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
        save_dir: str = "results/checkpoints",
        tensorboard_dir: str = "results/tensorboard",
        log_dir: str = "results/logs",
        run_name: Optional[str] = None,
        accelerator=None,
        artifact_prefix: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator is not None else device
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.artifact_prefix = artifact_prefix or run_name or "model"

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = (
            SummaryWriter(log_dir=os.path.join(tensorboard_dir, run_name or "run"))
            if accelerator is None or accelerator.is_main_process
            else None
        )
        self.logger = setup_logger(run_name or __name__, log_dir=log_dir)

        # log model size
        try:
            total, trainable = print_model_size(
                self.model, model_name=run_name or "Model"
            )
            self.logger.info(f"Model params: total={total:,}, trainable={trainable:,}")
        except Exception:
            # best-effort only; show exception details for debugging
            self.logger.exception("print_model_size failed")

    def save_checkpoint(
        self, epoch: int, name: str = "latest_model.pt", extra: Optional[Dict] = None
    ):
        if not name.startswith(f"{self.artifact_prefix}_"):
            name = f"{self.artifact_prefix}_{name}"
        path = os.path.join(self.save_dir, name)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            model_state = self.accelerator.get_state_dict(self.model)
        else:
            model_state = self.model.state_dict()
        state = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None:
            state["scaler_state"] = scaler.state_dict()
        if extra:
            state.update(extra)
        if self.accelerator is None or self.accelerator.is_main_process:
            temporary_path = path + ".tmp"
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
            try:
                self.logger.info(f"Saved checkpoint: {path}")
            except Exception as e:
                self.logger.warning("Failed logging checkpoint save: %s", e)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return path

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None,
    ):
        ckpt = _safe_torch_load(
            path,
            map_location=map_location or self.device,
        )

        model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, RuntimeError) as e:
                self.logger.warning(
                    "Failed to load optimizer state: %s",
                    e,
                )

        if self.scheduler is not None and "scheduler_state" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except (ValueError, RuntimeError) as e:
                self.logger.warning(
                    "Failed to load scheduler state: %s",
                    e,
                )
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])

        try:
            self.logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            self.logger.warning(
                "Failed logging checkpoint load: %s",
                e,
            )

        return ckpt

    def evaluate(self, dataloader: DataLoader) -> Dict:
        # Backwards-compatible wrapper that evaluates `self.model` on `self.device`.
        return self.evaluate_model(self.model, dataloader, device=self.device)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict:
        device = device or self.device
        if self.accelerator is None:
            model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_valid_tokens = 0
        total_correct = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                autocast = (
                    self.accelerator.autocast()
                    if self.accelerator is not None
                    else nullcontext()
                )
                with autocast:
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels, inputs)
                mask = labels != -100
                valid_count = int(mask.sum().item())
                loss_stats = torch.tensor(
                    [loss.detach().item() * valid_count, valid_count],
                    dtype=torch.float64,
                    device=device,
                )
                if self.accelerator is not None:
                    loss_stats = self.accelerator.reduce(loss_stats, reduction="sum")
                total_loss += loss_stats[0].item()

                unwrapped_model = (
                    self.accelerator.unwrap_model(model)
                    if self.accelerator is not None
                    else model
                )
                if getattr(unwrapped_model, "use_crf", False):
                    autocast = (
                        self.accelerator.autocast()
                        if self.accelerator is not None
                        else nullcontext()
                    )
                    with autocast:
                        decoded = unwrapped_model.decode(inputs, mask=mask)
                    batch_preds = torch.full_like(labels, -100)
                    for row, pred_seq in enumerate(decoded):
                        positions = torch.where(mask[row])[0]
                        if len(pred_seq) != len(positions):
                            raise RuntimeError(
                                "CRF path length does not match active labels"
                            )
                        batch_preds[row, positions] = torch.as_tensor(
                            pred_seq, dtype=labels.dtype, device=device
                        )
                else:
                    batch_preds = torch.argmax(outputs, dim=-1)

                if self.accelerator is not None:
                    batch_preds, metric_labels = self.accelerator.gather_for_metrics(
                        (batch_preds, labels)
                    )
                else:
                    metric_labels = labels
                metric_mask = metric_labels != -100
                total_valid_tokens += int(metric_mask.sum().item())
                total_correct += int(
                    ((batch_preds == metric_labels) & metric_mask).sum().item()
                )
                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(metric_labels.cpu().tolist())

        avg_loss = total_loss / max(1, total_valid_tokens)
        accuracy = (
            total_correct / max(1, total_valid_tokens)
            if total_valid_tokens > 0
            else 0.0
        )

        entity_f1 = compute_entity_level_metrics(all_preds, all_labels, ID2LABEL)[
            "overall"
        ]["f1-score"]
        return {"loss": avg_loss, "accuracy": accuracy, "entity_f1": entity_f1}

    def on_train_start(self):
        pass

    def on_train_end(self, results):
        return results

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        autocast = (
            self.accelerator.autocast()
            if self.accelerator is not None
            else nullcontext()
        )
        with autocast:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels, inputs)

        return loss, outputs, labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        grad_clip: Optional[float] = None,
        early_stop: Optional[int] = None,
        log_interval: int = 50,
        start_epoch: int = 1,
    ) -> Dict:
        self.on_train_start()

        best_entity_f1 = None
        best_path = None
        no_improve = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_entity_f1": [],
        }
        for epoch in range(start_epoch, epochs + 1):

            self.model.train()
            epoch_loss = 0.0
            total_valid_tokens = 0

            train_correct = 0
            train_valid_tokens = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}",
                disable=(
                    self.accelerator is not None
                    and not self.accelerator.is_local_main_process
                ),
            )

            for step, batch in enumerate(progress):

                self.optimizer.zero_grad()

                loss, outputs, labels = self.training_step(batch)

                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                if grad_clip:
                    if self.accelerator is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )

                self.optimizer.step()

                base_scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
                if self.scheduler is not None and not isinstance(
                    base_scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

                valid_count = int((labels != -100).sum().item())
                epoch_loss += loss.item() * valid_count
                total_valid_tokens += valid_count
                # Auxiliary token accuracy uses emission argmax (not CRF Viterbi).
                preds = torch.argmax(outputs, dim=-1)
                mask = labels != -100
                if preds.shape == labels.shape:
                    train_correct += ((preds == labels) & mask).sum().item()
                    train_valid_tokens += mask.sum().item()

                if self.writer is not None and step % log_interval == 0:

                    global_step = (epoch - 1) * len(train_loader) + step

                    self.writer.add_scalar(
                        "train/batch_loss",
                        loss.item(),
                        global_step,
                    )

                progress.set_postfix(loss=f"{loss.item():.4f}")

            progress.close()

            if self.accelerator is not None:
                totals = torch.tensor(
                    [epoch_loss, total_valid_tokens, train_correct, train_valid_tokens],
                    dtype=torch.float64,
                    device=self.device,
                )
                totals = self.accelerator.reduce(totals, reduction="sum").tolist()
                epoch_loss, total_valid_tokens, train_correct, train_valid_tokens = (
                    totals
                )

            avg_epoch_loss = epoch_loss / max(1, total_valid_tokens)
            train_acc = (
                train_correct / max(1, train_valid_tokens)
                if train_valid_tokens > 0
                else 0.0
            )

            history["train_loss"].append(avg_epoch_loss)
            history["train_acc"].append(train_acc)

            if self.writer is not None:
                self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
                self.writer.add_scalar("train/epoch_acc", train_acc, epoch)
            if self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(
                    f"Epoch {epoch} train_loss={avg_epoch_loss:.4f}, train_acc={train_acc:.4f}"
                )

            # Evaluate on validation set
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss", float("inf"))
                val_acc = val_metrics.get("accuracy", 0.0)
                val_entity_f1 = val_metrics.get("entity_f1", 0.0)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_entity_f1"].append(val_entity_f1)

                if self.writer is not None:
                    self.writer.add_scalar("val/loss", val_loss, epoch)
                    self.writer.add_scalar("val/acc", val_acc, epoch)
                    self.writer.add_scalar("val/entity_f1", val_entity_f1, epoch)
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info(
                        f"Epoch {epoch} val_loss={val_loss:.4f}, token_acc={val_acc:.4f}, "
                        f"entity_f1={val_entity_f1:.4f}"
                    )

            # Learning rate scheduler step handling
            if self.scheduler is not None:
                try:
                    base_scheduler = getattr(
                        self.scheduler, "scheduler", self.scheduler
                    )
                    if isinstance(base_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics is not None:
                            self.scheduler.step(val_loss)
                        else:
                            self.logger.warning(
                                "ReduceLROnPlateau requires val_loader. Scheduler skipped."
                            )
                except Exception as e:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        self.logger.warning(f"Scheduler.step() failed: {e}")

                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.writer is not None:
                    self.writer.add_scalar("train/lr", current_lr, epoch)

            extra = {
                "val_loss": val_metrics.get("loss") if val_metrics else None,
                "val_entity_f1": val_metrics.get("entity_f1") if val_metrics else None,
            }
            path = self.save_checkpoint(epoch, name="latest_model.pt", extra=extra)

            current_score = (
                val_metrics.get("entity_f1", 0.0) if val_metrics else -avg_epoch_loss
            )

            if best_entity_f1 is None or current_score > best_entity_f1:
                best_entity_f1 = current_score
                no_improve = 0
                best_path = os.path.join(
                    self.save_dir, f"{self.artifact_prefix}_best_model.pt"
                )
                try:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        shutil.copy2(path, best_path)
                        self.logger.info(
                            f"New best model saved at epoch {epoch} with selection_score {best_entity_f1:.4f}"
                        )
                    if self.accelerator is not None:
                        self.accelerator.wait_for_everyone()
                except Exception as e:
                    if self.accelerator is None or self.accelerator.is_main_process:
                        self.logger.warning(f"Failed to copy best model: {e}")
                    best_path = path
            else:
                no_improve += 1

            if early_stop and no_improve >= early_stop:
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info("Early stopping triggered")
                break

        results = {
            "best_path": best_path,
            "best_entity_f1": best_entity_f1 if val_loader is not None else None,
            "history": history,
        }
        results = self.on_train_end(results)
        if self.writer is not None:
            self.writer.close()
        return results


class DistillationTrainer(BaseTrainer):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        alpha: float = 0.5,
        temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(student_model, optimizer, criterion, device, **kwargs)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.kldiv = nn.KLDivLoss(reduction="none")

    def _distillation_loss(self, student_logits, teacher_logits, labels, input_ids):
        T = self.temperature
        p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
        p_t = nn.functional.softmax(teacher_logits / T, dim=-1)

        # Distill only the same first-subword positions used for supervised CRF/CE.
        attention_mask = (labels != -100).to(student_logits.dtype)

        # Compute KL divergence per token
        kd_loss_per_token = self.kldiv(p_s, p_t).sum(dim=-1) * (T * T)

        # Apply attention_mask and average over valid tokens
        kd_loss = (kd_loss_per_token * attention_mask).sum() / torch.clamp(
            attention_mask.sum(), min=1e-8
        )

        ce_loss = self.criterion(student_logits, labels, input_ids)

        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        student_logits = self.model(inputs)

        # Pass inputs into the loss function to compute an attention mask when needed
        loss = self._distillation_loss(student_logits, teacher_logits, labels, inputs)

        return loss, student_logits, labels
