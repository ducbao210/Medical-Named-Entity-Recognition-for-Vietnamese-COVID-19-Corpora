import random
from collections import defaultdict
from src.utils import setup_logger

logger = setup_logger(__name__)


class NERDataAugmentor:
    """
    Data augmentation utility for Named Entity Recognition (NER) tasks.
    Supports entity-level replacement augmentation.
    """

    def __init__(self, sentences, label2id, seed=42):
        """
        Parameters
        ----------
        sentences : list of dict
            Each element is in the format:
            {"words": [...], "tags": [...]}

        label2id : dict
            Mapping from label strings to integer IDs (not used directly here,
            but kept for compatibility with training pipelines).
        """
        self.sentences = sentences
        self.label2id = label2id
        self.rng = random.Random(seed)
        for index, sentence in enumerate(sentences):
            if len(sentence["words"]) != len(sentence["tags"]):
                raise ValueError(f"Sentence {index} has {len(sentence['words'])} words but {len(sentence['tags'])} tags")
        self.entity_dict = self._build_entity_dictionary()

    def _build_entity_dictionary(self):
        """
        Extract and group named entities by their entity type
        from the original dataset.
        """
        entity_dict = defaultdict(list)

        for sentence in self.sentences:
            words = sentence["words"]
            tags = sentence["tags"]

            current_entity_words = []
            current_entity_type = None

            for word, tag in zip(words, tags):
                if tag.startswith("B-"):
                    if current_entity_words:
                        entity_dict[current_entity_type].append(current_entity_words)

                    current_entity_type = tag[2:]
                    current_entity_words = [word]

                elif tag.startswith("I-") and current_entity_type == tag[2:]:
                    current_entity_words.append(word)

                else:
                    if current_entity_words:
                        entity_dict[current_entity_type].append(current_entity_words)
                        current_entity_words = []
                        current_entity_type = None

            if current_entity_words:
                entity_dict[current_entity_type].append(current_entity_words)

        # Remove duplicates
        for k in entity_dict:
            unique_entities = sorted(set(tuple(x) for x in entity_dict[k]))
            entity_dict[k] = [list(x) for x in unique_entities]

        return entity_dict

    def augment_sentence(self, sentence, replace_prob=0.3):
        """
        Create a new augmented sentence by randomly replacing entities
        with other entities of the same type.

        Parameters
        ----------
        sentence : dict
            {"words": [...], "tags": [...]}

        replace_prob : float
            Probability of replacing each entity.

        Returns
        -------
        dict
            Augmented sentence with the same structure.
        """
        words = sentence["words"].copy()
        tags = sentence["tags"].copy()

        new_words = []
        new_tags = []

        i = 0
        while i < len(words):
            tag = tags[i]

            if tag.startswith("B-") and self.rng.random() < replace_prob:
                entity_type = tag[2:]

                # Determine entity span length
                j = i + 1
                while j < len(words) and tags[j] == f"I-{entity_type}":
                    j += 1

                # Replace with another entity of the same type
                if (
                    entity_type in self.entity_dict
                    and len(self.entity_dict[entity_type]) > 1
                ):
                    replacement = self.rng.choice(self.entity_dict[entity_type])

                    new_words.extend(replacement)
                    new_tags.append(f"B-{entity_type}")
                    new_tags.extend([f"I-{entity_type}"] * (len(replacement) - 1))
                else:
                    # If no replacement available, keep original
                    new_words.extend(words[i:j])
                    new_tags.extend(tags[i:j])

                i = j
            else:
                new_words.append(words[i])
                new_tags.append(tags[i])
                i += 1

        if len(new_words) != len(new_tags):
            raise RuntimeError("Augmentation produced misaligned words and tags")
        return {"words": new_words, "tags": new_tags}

    def generate_augmented_dataset(self, multiplier=1, replace_prob=0.3):
        """
        Generate an augmented dataset by duplicating and augmenting sentences.

        Parameters
        ----------
        multiplier : int
            Number of augmented samples generated per original sentence.

        replace_prob : float
            Probability of entity replacement.

        Returns
        -------
        list of dict
            Augmented dataset including original + synthetic samples.
        """
        augmented_data = []

        for sentence in self.sentences:
            # Keep original sentence
            augmented_data.append(sentence)

            for _ in range(multiplier):
                aug_sent = self.augment_sentence(sentence, replace_prob)

                # Only add if there is a change (avoid exact duplicates)
                if aug_sent["words"] != sentence["words"]:
                    augmented_data.append(aug_sent)

        logger.info(
            f"Augmentation completed: {len(self.sentences)} -> {len(augmented_data)} samples."
        )

        return augmented_data
