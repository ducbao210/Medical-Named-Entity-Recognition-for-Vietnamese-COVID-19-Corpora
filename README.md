# Vietnamese COVID-19 Named Entity Recognition

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)

Vietnamese NER experiments on the [PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19) corpus. The project trains and evaluates sequence taggers for identifying people, locations, dates, diseases, and other entities in Vietnamese COVID-19 reports. It includes recurrent and Transformer baselines, pretrained encoders, an optional CRF decoding layer, and PhoBERT-to-recurrent/Transformer knowledge distillation.

## Overview

The dataset uses word-segmented CoNLL files with BIO labels. The configuration defines 20 tags: `O` plus `B-`/`I-` tags for 10 entity types (organization, symptom/disease, location, date, patient ID, age, name, job, transportation, and gender).

Training reads the train and development splits, applies the repository's training-set augmentation, aligns word labels with tokenizer subwords, and selects the best checkpoint using validation entity-level F1. Evaluation uses the test split and reports token-level and BIO entity-level metrics. Inference accepts raw Vietnamese text, word-segments it with `underthesea`, and prints the predicted tag for each word.

Supported architectures:

- **Recurrent baselines:** LSTM and BiLSTM.
- **Transformer baseline:** a Transformer trained from scratch.
- **Pretrained encoders:** PhoBERT, PhoBERT with LoRA, XLM-RoBERTa, and ViDeBERTa.
- **Optional CRF:** a linear-chain CRF decoder can be enabled for each model with `--use_crf`.

## Repository structure

```text
.
├── data/                         # train/dev/test word-level CoNLL files
├── notebook/
│   ├── eda.ipynb                 # exploratory data analysis
│   ├── ner-kaggle-training.ipynb # training/eval/inference and distillation, no CRF
│   └── crf-ver-ner-kaggle-training.ipynb
├── src/
│   ├── checkpoints/              # checkpoint lookup and upload helpers
│   ├── config/                   # data paths, labels, model/KD settings
│   ├── data/                     # CoNLL parser, tokenizer alignment, augmentation
│   ├── models/                   # model architectures and CRF layer
│   ├── pipeline/                 # train, evaluate, inference, and distill workflows
│   ├── training/                 # base and distillation trainers
│   └── utils/                    # metrics, plots, prediction and logging helpers
├── results/                      # generated checkpoints, metrics, plots, logs, predictions
├── download_data.py              # fetches CoNLL files from the upstream corpus
├── main.py                       # command-line entry point
├── requirements.txt              # standard Python dependencies
└── requirements-kaggle.txt       # Kaggle dependency list
```

`results/` and the local dataset are git-ignored; training and evaluation create their output directories as needed.

## Prerequisites and setup

- Python 3.x.
- PyTorch installed with a build appropriate for your CPU/GPU. PyTorch is not pinned in the requirements files; install it for your environment before installing the remaining dependencies.
- Internet access to download the dataset and pretrained Hugging Face model/tokenizer files. A GPU is recommended for pretrained model training but is not required by the CLI.
- A Hugging Face write token and repository ID only if you want to upload checkpoints.

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Download the CoNLL splits into `data/`:

```bash
python download_data.py
```

The downloader fetches `.conll` files from the upstream repository's `data/word` directory. Confirm that `data/train_word.conll`, `data/dev_word.conll`, and `data/test_word.conll` are present before running a workflow.

Checkpoint upload is optional. To configure it, copy `.env.example` to `.env`, then set your own `HF_TOKEN` and `HF_CHECKPOINT_REPO_ID` values there. Do not commit `.env`.

## Train

Train all supported models without CRF:

```bash
python main.py --mode train --model all
```

Train all supported models with CRF:

```bash
python main.py --mode train --model all --use_crf
```

Select one model with `--model`, for example:

```bash
python main.py --mode train --model phobert --use_crf
python main.py --mode train --model bilstm
```

Training uses the train and development files. Model defaults such as learning rate, batch size, maximum sequence length, epochs, and early-stopping patience are in `src/config/settings.py`; common overrides include `--epochs`, `--batch_size`, and `--lr`.

To upload a completed checkpoint to the configured Hugging Face repository, add `--upload_checkpoint`:

```bash
python main.py --mode train --model phobert --use_crf --upload_checkpoint
```

## Knowledge distillation

The two training notebooks run distillation with a PhoBERT teacher and a BiLSTM student. The CLI also supports LSTM and scratch Transformer students. The teacher is loaded from the matching PhoBERT checkpoint (with or without CRF), and the student shares the PhoBERT tokenizer so their token-level outputs align. The distillation trainer combines the teacher's softened predictions with the supervised NER objective; the default temperature is `4.0` and the configured distillation weight is `0.7` (`KDConfig` in `src/config/settings.py`).

Train the notebook's BiLSTM student without CRF:

```bash
python main.py --mode distill --model bilstm
```

Train with CRF, matching the CRF notebook:

```bash
python main.py --mode distill --model bilstm --use_crf
```

The notebook workflow evaluates the distilled student using its saved checkpoint. For example:

```bash
python main.py --mode evaluate --model bilstm \
  --checkpoint results/checkpoints/bilstm_distilled_best_model.pt

python main.py --mode evaluate --model bilstm --use_crf \
  --checkpoint results/checkpoints/bilstm_distilled_crf_best_model.pt
```

To distill multiple supported students, use `--model all` with `--mode distill`. Only `lstm`, `bilstm`, and `transformer` are valid students for the current PhoBERT teacher/tokenizer setup.

## Evaluate

Evaluate all model checkpoints on the test split:

```bash
python main.py --mode evaluate --model all
python main.py --mode evaluate --model all --use_crf
```

Or evaluate one checkpoint:

```bash
python main.py --mode evaluate --model phobert --use_crf \
  --checkpoint results/checkpoints/phobert_crf_best_model.pt
```

Evaluation writes token-level metrics to `results/metrics/`, predictions to `results/predictions/`, and confusion matrices/entity-distribution plots to `results/plots/`. The entity-level BIO report is printed to the run log. Training checkpoints are selected by validation entity F1; token accuracy and token-level F1 are separate metrics.

## Inference

Pass one Vietnamese sentence with `--infer_text` and point to a checkpoint. The model and `--use_crf` setting must match the checkpoint:

```bash
python main.py --mode infer --model bilstm \
  --checkpoint results/checkpoints/bilstm_distilled_best_model.pt \
  --infer_text "Bệnh nhân 47 tuổi được đưa đến bệnh viện tại Hà Nội."
```

CRF example:

```bash
python main.py --mode infer --model bilstm --use_crf \
  --checkpoint results/checkpoints/bilstm_distilled_crf_best_model.pt \
  --infer_text "Bệnh nhân 47 tuổi được đưa đến bệnh viện tại Hà Nội."
```

Inference highlights non-`O` predictions in the terminal. The current CLI accepts a text string; it does not expose a file-input option.

## Recorded results

The following scores are read from the `results/metrics/*_token_metrics.csv` files present in this workspace. They are **token-level** test metrics, not span/entity F1. `Macro F1` averages across the reported labels, while `Weighted F1` weights each label by its support. Bold marks the best recorded value in each column.

| Model | CRF | Token accuracy | Macro F1 | Weighted F1 |
|---|:---:|---:|---:|---:|
| LSTM | No | 95.12% | 74.47% | 94.96% |
| LSTM | Yes | 95.33% | 74.61% | 95.16% |
| BiLSTM | No | 95.91% | 70.52% | 95.70% |
| BiLSTM | Yes | 96.30% | 75.57% | 96.19% |
| Transformer (scratch) | No | 94.81% | 74.01% | 94.70% |
| Transformer (scratch) | Yes | 95.13% | 77.74% | 95.01% |
| PhoBERT | No | 98.12% | 86.25% | 98.10% |
| PhoBERT | Yes | 98.15% | 87.42% | 98.13% |
| PhoBERT + LoRA | No | 98.21% | 87.68% | 98.19% |
| **PhoBERT + LoRA** | **Yes** | **98.29%** | **88.54%** | **98.26%** |
| XLM-RoBERTa | No | 97.35% | 82.76% | 97.31% |
| XLM-RoBERTa | Yes | 97.46% | 77.70% | 97.42% |
| ViDeBERTa | No | 87.32% | 47.29% | 86.28% |
| ViDeBERTa | Yes | 87.38% | 47.61% | 86.43% |

### Notes on the results

- PhoBERT with LoRA and CRF has the strongest recorded token-level scores in these CSVs. CRF improves token accuracy and weighted F1 slightly for most model pairs, though XLM-RoBERTa's macro F1 is higher without CRF in this snapshot.
- The macro scores are substantially below the weighted scores for several models. This gap, together with the large `O` support in the reports, indicates that aggregate accuracy/weighted F1 can hide weaker performance on less frequent entity tags. Review per-entity BIO F1 and support before choosing a model.
- These CSVs are token reports. The repository's evaluation code also computes span-based entity F1, but the CSVs do not store it. The files also do not identify their checkpoint provenance; in particular, the notebooks evaluate a distilled BiLSTM using the same generic `bilstm` metric filenames. Treat the BiLSTM rows as recorded artifacts, not as a clean baseline-versus-distilled comparison.
- The table is a snapshot of artifacts already in the repository, not the result of a fresh training/evaluation run. Re-run evaluation to regenerate metrics for a specific checkpoint.

## Citation

The dataset/model task is described in:

```bibtex
@inproceedings{PhoNER_COVID19,
  title     = {{COVID-19 Named Entity Recognition for Vietnamese}},
  author    = {Thinh Hung Truong and Mai Hoang Dao and Dat Quoc Nguyen},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2021}
}
```
