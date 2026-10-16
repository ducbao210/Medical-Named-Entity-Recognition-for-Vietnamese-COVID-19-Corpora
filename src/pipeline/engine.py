import os
import logging
import torch
import torch.optim as optim
from underthesea import word_tokenize

from src.config import (
    TRAIN_FILE, DEV_FILE, TEST_FILE, LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS,
    LoRAConfig, get_model_dirs,
)
from src.data import read_conll, VietnameseNERDataset, get_dataloader, encode_words_with_alignment, NERDataAugmentor
from src.models import (get_tokenizer, get_model, PhoBERTModel, NERLoss)
from src.training import BaseTrainer, DistillationTrainer
from src.utils import (setup_logger, compute_metrics, compute_entity_level_metrics,
                       plot_loss_curve, plot_acc_curve, plot_confusion_matrix,
                       plot_entity_distribution, save_metrics_csv, save_predictions)
from src.checkpoints import ensure_checkpoint, upload_checkpoints, CheckpointError

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transformers

transformers.utils.logging.set_verbosity_error()  # Suppress non-essential transformers logs

logger = setup_logger("engine")


def _load_checkpoint_file(path, map_location, *, allow_full_pickle=False):
    load_kwargs = {"map_location": map_location}
    if allow_full_pickle:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _load_model_for_eval(model, chk_path, device, logger):

    ckpt = _load_checkpoint_file(chk_path, map_location=device, allow_full_pickle=False)
    model_state = ckpt.get("model_state", ckpt)

    # Fail fast: evaluating with random/missing adapter or CRF weights is invalid.
    model.load_state_dict(model_state, strict=True)

    model.eval()
    return model.to(device), device


def run_evaluation(
    model, dataloader, tokenizer, id2label, dataset_sentences, save_pred_path=None
):
    """Run rigorous token-level and entity-level evaluation."""
    model.eval()
    all_preds = []
    all_labels = []

    device = next(model.parameters()).device
    logger.info("Running evaluation cycle...")

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            if hasattr(model, "use_crf") and model.use_crf:
                # CRF models return Viterbi paths as lists
                batch_preds = model.decode(inputs, mask=(labels != -100))
                for preds, gold_labels in zip(batch_preds, labels.cpu().tolist()):
                    aligned = [-100] * len(gold_labels)
                    positions = [
                        i for i, label in enumerate(gold_labels) if label != -100
                    ]
                    if len(preds) != len(positions):
                        raise RuntimeError(
                            "CRF path length does not match active gold positions"
                        )
                    for pos, pred in zip(positions, preds):
                        aligned[pos] = pred
                    all_preds.append(aligned)
                    all_labels.append(gold_labels)
            else:
                # Non-CRF standard prediction
                logits = model(inputs)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                for p, g in zip(preds, labels.cpu().tolist()):
                    all_preds.append(p)
                    all_labels.append(g)

    # Flatten for token-level metrics
    flat_preds = []
    flat_labels = []
    for p_seq, l_seq in zip(all_preds, all_labels):
        for p, l in zip(p_seq, l_seq):
            flat_preds.append(p)
            flat_labels.append(l)

    # Compute Token-Level Classification Report
    token_report = compute_metrics(flat_preds, flat_labels, LABEL_LIST)

    # Compute Entity-Level precision, recall, and F1 (BIO span-based evaluation)
    entity_report = compute_entity_level_metrics(all_preds, all_labels, id2label)

    # Save predictions to file if requested
    if save_pred_path and dataset_sentences:
        tokens_list = [sent["words"] for sent in dataset_sentences]
        # Align prediction tags with tokens (excluding paddings)
        clean_trues = []
        clean_preds = []
        for p_seq, l_seq in zip(all_preds, all_labels):
            t_seq_clean = [l for l in l_seq if l != -100]
            p_seq_clean = [p for p, l in zip(p_seq, l_seq) if l != -100]

            clean_trues.append(t_seq_clean)
            clean_preds.append(p_seq_clean)

        save_predictions(
            tokens_list, clean_trues, clean_preds, save_pred_path, id2label
        )

    return token_report, entity_report, flat_preds, flat_labels, all_preds, all_labels


def run_train(
    args,
    bert_config,
    tf_config,
    lstm_config,
    tokenizer,
    device,
    LABEL2ID,
    accelerator=None,
):
    logger.info("Loading training and validation datasets...")

    # Read raw CoNLL-formatted data files
    train_sentences = read_conll(TRAIN_FILE)
    val_sentences = read_conll(DEV_FILE)

    # Apply data augmentation to the training set
    logger.info("Applying Data Augmentation...")
    augmentor = NERDataAugmentor(train_sentences, LABEL2ID)
    augmented_train_sentences = augmentor.generate_augmented_dataset(
        multiplier=1, replace_prob=0.3
    )

    # Determine which models to train
    if "all" in args.model:
        models_to_train = [
            "lstm",
            "bilstm",
            "transformer",
            "phobert",
            "phobert-lora",
            "xlm-roberta",
            "videberta",
        ]
    else:
        models_to_train = args.model

    # Loop over models to train them sequentially
    for current_model in models_to_train:
        model_tokenizer = get_tokenizer(current_model, fallback=tokenizer)
        logger.info(
            f"\n{'='*50}\nSTART TRAINING WITH MODEL: {current_model.upper()}\n{'='*50}"
        )

        if current_model in ("phobert", "phobert-lora", "xlm-roberta", "videberta"):
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        current_max_len = active_cfg.max_seq_length
        current_batch_size = args.batch_size or active_cfg.batch_size
        current_val_batch_size = active_cfg.val_batch_size
        current_epochs = args.epochs or active_cfg.epochs
        current_lr = args.lr or active_cfg.learning_rate

        # Initialize Dataset & DataLoader according to the active model configuration
        train_dataset = VietnameseNERDataset(
            augmented_train_sentences, model_tokenizer, current_max_len, LABEL2ID
        )
        val_dataset = VietnameseNERDataset(
            val_sentences, model_tokenizer, current_max_len, LABEL2ID
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        logger.info(f"Initializing {current_model} model for training...")
        model = get_model(
            current_model,
            model_tokenizer.vocab_size,
            pad_token_id=model_tokenizer.pad_token_id,
            use_crf=args.use_crf,
        )
        model = model.to(device)
        criterion = NERLoss(model)
        if current_model == "phobert-lora":
            lora_params, classifier_params, crf_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "classifier" in name:
                    classifier_params.append(param)
                elif "crf" in name:
                    crf_params.append(param)
                else:
                    lora_params.append(param)

            param_groups = [
                {"params": lora_params, "lr": 2e-4},
                {"params": classifier_params, "lr": 1e-3},
            ]
            if crf_params:
                param_groups.append({"params": crf_params, "lr": 1e-3})
            optimizer = optim.AdamW(param_groups)
            total_steps = len(train_loader) * current_epochs
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr
            )

        if accelerator is not None:
            model, optimizer, train_loader, val_loader = accelerator.prepare(
                model, optimizer, train_loader, val_loader
            )

        if current_model == "phobert-lora":
            from transformers import get_linear_schedule_with_warmup

            warmup_steps = int(total_steps * LoRAConfig().warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        if accelerator is not None:
            scheduler = accelerator.prepare(scheduler)

        dirs = get_model_dirs(current_model, args.use_crf)
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            save_dir=dirs["checkpoints"],
            tensorboard_dir=dirs["tensorboard"],
            log_dir=dirs["logs"],
            run_name=f"{current_model}_crf_{args.use_crf}",
            accelerator=accelerator,
            artifact_prefix=dirs["prefix"],
        )

        start_epoch = 1
        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info(f"Restoring state from checkpoint: {args.checkpoint}")
            ckpt = trainer.load_checkpoint(args.checkpoint)
            start_epoch = ckpt.get("epoch", 0) + 1

        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=current_epochs,
            early_stop=args.patience or active_cfg.patience,
            grad_clip=1.0,
            start_epoch=start_epoch,
        )
        logger.info(
            f"Training completed. Best model for {current_model} saved at: {results['best_path']}"
        )
        if args.upload_checkpoint:
            if not args.hf_repo_id:
                raise CheckpointError("--upload_checkpoint requires --hf_repo_id or HF_CHECKPOINT_REPO_ID")
            upload_checkpoints(
                repo_id=args.hf_repo_id, model_name=current_model,
                use_crf=args.use_crf, token=args.hf_token,
            )

        history = results.get("history")
        if history:
            logger.info(f"Generating training plots for {current_model}...")
            plot_loss_curve(
                history,
                save_path=os.path.join(
                    dirs["plots"], f"{dirs['prefix']}_loss_curve.png"
                ),
            )
            plot_acc_curve(
                history,
                save_path=os.path.join(dirs["plots"], f"{dirs['prefix']}_acc_curve.png"),
            )


def run_evaluate(
    args,
    bert_config,
    tf_config,
    lstm_config,
    tokenizer,
    device,
    LABEL2ID,
    ID2LABEL,
    LABEL_LIST,
):
    logger.info("Loading evaluation (test) dataset...")
    test_sentences = read_conll(TEST_FILE)

    if "all" in args.model:
        models_to_eval = [
            "lstm",
            "bilstm",
            "transformer",
            "phobert",
            "phobert-lora",
            "xlm-roberta",
            "videberta",
        ]
    else:
        models_to_eval = args.model

    for current_model in models_to_eval:
        model_tokenizer = get_tokenizer(current_model, fallback=tokenizer)
        logger.info(
            f"\n{'='*50}\nSTART EVALUATING WITH MODEL: {current_model.upper()}\n{'='*50}"
        )

        if current_model in ("phobert", "phobert-lora", "xlm-roberta", "videberta"):
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        test_loader = get_dataloader(
            TEST_FILE,
            model_tokenizer,
            active_cfg.val_batch_size,
            active_cfg.max_seq_length,
            LABEL2ID,
            shuffle=False,
        )

        model = get_model(
            current_model,
            model_tokenizer.vocab_size,
            pad_token_id=model_tokenizer.pad_token_id,
            use_crf=args.use_crf,
        )
        dirs = get_model_dirs(current_model, args.use_crf)

        try:
            chk_path = str(ensure_checkpoint(
                current_model,
                args.use_crf,
                requested=args.checkpoint,
                repo_id=args.hf_repo_id,
                token=args.hf_token,
            ))
        except CheckpointError as exc:
            logger.error(f"Could not resolve checkpoint for {current_model}: {exc}")
            continue

        logger.info(f"Loading weights from checkpoint: {chk_path}")
        try:
            model, eval_device = _load_model_for_eval(model, chk_path, device, logger)
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{chk_path}': {e}")
            continue

        pred_save_path = os.path.join(dirs["predictions"], f"{dirs['prefix']}_predictions.txt")
        token_report, entity_report, flat_preds, flat_labels, _, _ = run_evaluation(
            model,
            test_loader,
            model_tokenizer,
            ID2LABEL,
            test_sentences,
            save_pred_path=pred_save_path,
        )

        logger.info(f"\nToken-Level Evaluation Report ({current_model.upper()}):")
        logger.info(f"Token accuracy (not entity F1): {token_report['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {token_report['macro avg']['f1-score']:.4f}")
        logger.info(f"\nEntity-Level BIO Evaluation Report ({current_model.upper()}):")
        logger.info(f"Overall F1-Score: {entity_report['overall']['f1-score']:.4f}")
        logger.info(
            f"\nDetailed Entity Report ({current_model.upper()}):\n{entity_report['report']}"
        )
        csv_path = os.path.join(dirs["metrics"], f"{dirs['prefix']}_token_metrics.csv")
        save_metrics_csv(token_report, csv_path)

        plot_confusion_matrix(
            flat_preds,
            flat_labels,
            LABEL_LIST,
            os.path.join(dirs["plots"], f"{dirs['prefix']}_confusion_matrix.png"),
        )
        plot_entity_distribution(
            flat_labels,
            LABEL_LIST,
            os.path.join(dirs["plots"], f"{dirs['prefix']}_entity_distribution.png"),
        )


def run_infer(args, bert_config, tf_config, lstm_config, tokenizer, device, ID2LABEL):
    """Run inference supporting all model variants including the pure Transformer."""
    if not args.infer_text:
        logger.error("Error: --infer_text is required when --mode is set to 'infer'.")
        return

    if "all" in args.model:
        models_to_infer = [
            "lstm",
            "bilstm",
            "transformer",
            "phobert",
            "phobert-lora",
            "xlm-roberta",
            "videberta",
        ]
    else:
        models_to_infer = args.model

    for current_model in models_to_infer:
        model_tokenizer = get_tokenizer(current_model, fallback=tokenizer)
        logger.info(
            f"\n{'='*50}\nSTART INFERENCING WITH MODEL: {current_model.upper()}\n{'='*50}"
        )

        # Dynamically select model configuration to avoid mismatched max_length
        if current_model in ("phobert", "phobert-lora", "xlm-roberta", "videberta"):
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        max_seq_length = active_cfg.max_seq_length

        model = get_model(
            current_model,
            model_tokenizer.vocab_size,
            pad_token_id=model_tokenizer.pad_token_id,
            use_crf=args.use_crf,
        )
        dirs = get_model_dirs(current_model, args.use_crf)

        try:
            chk_path = str(ensure_checkpoint(
                current_model, args.use_crf, requested=args.checkpoint,
                repo_id=args.hf_repo_id, token=args.hf_token,
            ))
        except CheckpointError as exc:
            logger.error(f"Could not resolve checkpoint for {current_model}: {exc}")
            continue

        infer_device = device
        logger.info(f"Loading weights from {chk_path}")
        ckpt = _load_checkpoint_file(
            chk_path, map_location=infer_device, allow_full_pickle=False
        )
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)

        model = model.to(infer_device)
        model.eval()

        logger.info(f"Input sentence: {args.infer_text}")
        segmented_text = word_tokenize(args.infer_text, format="text")
        words = segmented_text.split()

        input_ids_list, word_ids = encode_words_with_alignment(
            model_tokenizer, words, max_seq_length, pad_to_max=True
        )
        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(infer_device)

        # For CRF, decode only first-subword positions, matching the training label mask.
        first_word_indices = []
        previous_word_idx = None
        for pos, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                first_word_indices.append(pos)
            previous_word_idx = word_idx
        active_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if first_word_indices:
            active_mask[0, first_word_indices] = True
        with torch.no_grad():
            decoded_output = (
                model.decode(input_ids, mask=active_mask)
                if getattr(model, "use_crf", False)
                else model.decode(input_ids)
            )
            if not decoded_output:
                logger.warning("Model returned empty decode output.")
                continue
            preds = decoded_output[0]

        word_preds = ["O"] * len(words)
        if getattr(model, "use_crf", False):
            for word_idx, tag_id in zip(
                [word_ids[i] for i in first_word_indices], preds
            ):
                if word_idx is not None and word_idx < len(words):
                    word_preds[word_idx] = ID2LABEL.get(tag_id, "O")
        else:
            for idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx < len(words):
                    if idx == 0 or word_ids[idx - 1] != word_idx:
                        word_preds[word_idx] = ID2LABEL.get(preds[idx], "O")

        logger.info(f"\n--- INFERENCE RESULTS ({current_model.upper()}) ---")
        for word, tag in zip(words, word_preds):
            if tag != "O":
                print(f"\033[92m{word} [{tag}]\033[0m")
            else:
                print(word)
        logger.info("-------------------------")


def run_distill(
    args,
    bert_config,
    tf_config,
    lstm_config,
    kd_config,
    tokenizer,
    device,
    LABEL2ID,
    NUM_LABELS,
):
    """Multi-architecture knowledge distillation: distill PhoBERT into LSTM/Transformer students."""
    logger.info("=== KNOWLEDGE DISTILLATION ===")

    if "all" in args.model:
        models_to_distill = ["lstm", "bilstm", "transformer"]
    else:
        models_to_distill = args.model
    allowed_students = {"lstm", "bilstm", "transformer"}
    unsupported_students = set(models_to_distill) - allowed_students
    if unsupported_students:
        raise ValueError(
            "Distillation students must share the PhoBERT tokenizer; "
            f"unsupported: {sorted(unsupported_students)}"
        )

    # The teacher model is fixed to the PhoBERT-base configuration
    teacher = PhoBERTModel(num_labels=NUM_LABELS, use_crf=args.use_crf)
    teacher_dirs = get_model_dirs("phobert", args.use_crf)

    try:
        chk_path = str(ensure_checkpoint(
            "phobert", args.use_crf, requested=args.checkpoint,
            repo_id=args.hf_repo_id, token=args.hf_token,
        ))
    except CheckpointError as exc:
        logger.error(f"Distillation requires a Teacher PhoBERT checkpoint: {exc}")
        return

    logger.info(f"Loading Teacher model weights from: {chk_path}")
    ckpt = _load_checkpoint_file(chk_path, map_location=device, allow_full_pickle=False)
    teacher.load_state_dict(ckpt.get("model_state", ckpt))
    teacher = teacher.to(device)

    for current_model in models_to_distill:
        logger.info(
            f"\n{'='*50}\nKNOWLEDGE DISTILLATION - STUDENT MODEL: {current_model.upper()}\n{'='*50}"
        )

        # Configure student-specific hyperparameters
        if current_model == "transformer":
            student_cfg = tf_config
        else:
            student_cfg = lstm_config

        max_seq_length = student_cfg.max_seq_length
        val_batch_size = student_cfg.val_batch_size
        batch_size = args.batch_size or student_cfg.batch_size
        epochs = args.epochs or student_cfg.epochs
        learning_rate = student_cfg.learning_rate

        train_loader = get_dataloader(
            TRAIN_FILE, tokenizer, batch_size, max_seq_length, LABEL2ID, shuffle=True
        )
        val_loader = get_dataloader(
            DEV_FILE, tokenizer, val_batch_size, max_seq_length, LABEL2ID, shuffle=False
        )

        student = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        student = student.to(device)

        # Use student-specific optimizer and hyperparameters as configured
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, student.parameters()), lr=learning_rate
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        criterion = NERLoss(student)

        student_dirs = get_model_dirs(f"{current_model}_distilled", args.use_crf)

        trainer = DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            alpha=kd_config.alpha,
            temperature=kd_config.temperature,
            save_dir=student_dirs["checkpoints"],
            tensorboard_dir=student_dirs["tensorboard"],
            log_dir=student_dirs["logs"],
            run_name=f"student_distilled_{current_model}_crf_{args.use_crf}",
            artifact_prefix=student_dirs["prefix"],
        )

        logger.info(f"Starting Knowledge Distillation for Student: {current_model}...")
        results = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stop=args.patience or student_cfg.patience,
            grad_clip=1.0,
        )
        logger.info(
            f"Distillation finished. Best checkpoint for {current_model} saved at: {results['best_path']}"
        )
        if args.upload_checkpoint:
            if not args.hf_repo_id:
                raise CheckpointError("--upload_checkpoint requires --hf_repo_id or HF_CHECKPOINT_REPO_ID")
            upload_checkpoints(
                repo_id=args.hf_repo_id, model_name=f"{current_model}_distilled",
                use_crf=args.use_crf, token=args.hf_token,
            )
