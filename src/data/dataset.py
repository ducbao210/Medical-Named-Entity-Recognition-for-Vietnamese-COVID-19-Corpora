import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import unicodedata


def read_conll(file_path):
    """
    Parse CoNLL/TXT formatted files for Named Entity Recognition.

    Sentences are separated by blank lines. Each non-empty line contains a word
    and its corresponding tag separated by space or tab.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.

    Returns
    ---
    list of dict
        A list where each item is a dictionary with:
        - "words": list of str (tokens in a sentence)
        - "tags": list of str (NER labels aligned with words)
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL file not found: {file_path}")

    sentences = []
    words, tags = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append({"words": words, "tags": tags})
                    words, tags = [], []
                continue

            parts = line.split()
            if len(parts) >= 2:
                word = unicodedata.normalize("NFC", parts[0])
                tag = unicodedata.normalize("NFC", parts[-1])

                words.append(word)
                tags.append(tag)

    if words:
        sentences.append({"words": words, "tags": tags})

    return sentences


def encode_words_with_alignment(tokenizer, words, max_length, pad_to_max=True):
    """Encode pre-segmented words, truncating only between whole words."""
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = list(encoded["input_ids"])
        word_ids = list(encoded.word_ids())
    else:
        content_ids, content_word_ids = [], []
        for word_id, word in enumerate(words):
            pieces = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(pieces)
            content_ids.extend(token_ids)
            content_word_ids.extend([word_id] * len(token_ids))
        input_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
        special_mask = tokenizer.get_special_tokens_mask(content_ids)
        word_ids = []
        content_index = 0
        for is_special in special_mask:
            if is_special:
                word_ids.append(None)
            else:
                word_ids.append(content_word_ids[content_index])
                content_index += 1
    if len(input_ids) != len(word_ids):
        raise ValueError("Tokenizer input_ids and word_ids have different lengths")

    content_positions = [i for i, word_id in enumerate(word_ids) if word_id is not None]
    if len(input_ids) > max_length:
        if not content_positions:
            raise ValueError(
                "max_length is too small to retain the tokenizer special tokens"
            )
        first_content, last_content = content_positions[0], content_positions[-1]
        prefix_positions = list(range(first_content))
        suffix_positions = list(range(last_content + 1, len(input_ids)))
        budget = max_length - len(prefix_positions) - len(suffix_positions)
        if budget < 0:
            raise ValueError("max_length is smaller than required special tokens")

        groups = []
        for position in content_positions:
            word_id = word_ids[position]
            if not groups or groups[-1][0] != word_id:
                groups.append((word_id, [position]))
            else:
                groups[-1][1].append(position)

        selected_positions = []
        used = 0
        for _, positions in groups:
            if used + len(positions) > budget:
                break
            selected_positions.extend(positions)
            used += len(positions)

        positions = prefix_positions + selected_positions + suffix_positions
        input_ids = [input_ids[i] for i in positions]
        word_ids = [word_ids[i] for i in positions]

    if pad_to_max:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id for fixed-length batches"
            )
        if len(input_ids) < max_length:
            padding = max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            word_ids.extend([None] * padding)
    if len(input_ids) > max_length:
        raise RuntimeError("Whole-word truncation failed to respect max_length")
    return input_ids, word_ids


class VietnameseNERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition (NER) using HuggingFace tokenizers.

    This dataset performs:
    - Encoding of pre-segmented words with fast or slow tokenizers
    - First-subword label alignment using word_ids or per-word tokenization
    - Padding/truncation to a fixed maximum length
    - Ignoring special and continuation subword positions with -100

    Parameters
    ---
    sentences : list of dict
        Output from read_conll(), each item contains:
        - "words": list of tokens
        - "tags": list of NER labels

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer (e.g. PhoBERT, BERT, XLM-R).

    max_len : int
        Maximum sequence length for padding/truncation.

    label2id : dict
        Mapping from label string to integer ID.

    Attributes
    ---
    ignore_index : int
        Label value used to ignore subword tokens in loss computation.
    """

    def __init__(self, sentences, tokenizer, max_len, label2id):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.ignore_index = -100

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words, tags = sentence["words"], sentence["tags"]
        if len(words) != len(tags):
            raise ValueError(
                f"Sentence {idx} has {len(words)} words but {len(tags)} tags"
            )
        unknown = sorted(set(tags) - set(self.label2id))
        if unknown:
            raise ValueError(f"Unknown NER tag(s) {unknown} in sentence {idx}")

        input_ids, word_ids = encode_words_with_alignment(
            self.tokenizer, words, self.max_len, pad_to_max=True
        )
        labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(self.ignore_index)
            elif word_id != previous_word_id:
                labels.append(self.label2id[tags[word_id]])
            else:
                labels.append(self.ignore_index)
            previous_word_id = word_id

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def get_dataloader(file_path, tokenizer, batch_size, max_len, label2id, shuffle=False):
    """
    Helper function to load a CoNLL file, create a VietnameseNERDataset,
    and return a PyTorch DataLoader.

    This function acts as a pipeline wrapper:
    CoNLL file → parsed sentences → Dataset → DataLoader.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer used for subword encoding.

    batch_size : int
        Number of samples per batch.

    max_len : int
        Maximum sequence length for padding/truncation.

    label2id : dict
        Mapping from label string to integer IDs.

    shuffle : bool
        Whether to shuffle the dataset each epoch.

    Returns
    ---
    DataLoader
        PyTorch DataLoader object ready for training or evaluation.
    """

    sentences = read_conll(file_path)
    dataset = VietnameseNERDataset(sentences, tokenizer, max_len, label2id)

    workers = 0 if os.name == "nt" else 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )

    return dataloader
