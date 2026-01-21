import os
import sys
import re
from abc import abstractmethod
from dataclasses import dataclass
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
from collections import Counter
from itertools import chain
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass



@dataclass(frozen=True)
class Vocabulary:
    """Immutable vocabulary mapping for text tokenization.

    Provides bidirectional mapping between tokens and their integer IDs.

    Attributes:
        id2token: List mapping token IDs to token strings.
        token2id: Dictionary mapping token strings to their IDs.
    """

    id2token: list[str]
    token2id: dict[str, int]

    @property
    def size(self):
        """Return the total number of tokens in the vocabulary."""
        return len(self.id2token)


class SplitterMixin:
    """Mixin providing data splitting functionality for train/validation/test sets."""

    def split(self, x: np.ndarray, y: np.ndarray, validation_split: float = 0.2, test_split: float = 0.0):
        """Split data into train, validation, and optionally test sets.

        Args:
            x: Feature data array.
            y: Target data array.
            validation_split: Fraction of data to use for validation (default: 0.2).
            test_split: Fraction of data to use for test set (default: 0.0).

        Returns:
            Tuple of (train_data, validation_data, test_data) where each is a tuple of (x, y).
            test_data is None if test_split is effectively 0.
        """
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(validation_split + test_split))
        if 1e-6 > test_split:
            train_data = (x_train, y_train)
            validation_data = (x_temp, y_temp)
            test_data = None
        else:
            val_size = validation_split / (validation_split + test_split)
            x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(1 - val_size))
            train_data = (x_train, y_train)
            validation_data = (x_val, y_val)
            test_data = (x_test, y_test)
        return train_data, validation_data, test_data


class Loader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement the load method to load data from various sources.
    """

    @abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and return feature and target data.

        Returns:
            Tuple of (features, targets) as numpy arrays.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")


class BaseDataset(Dataset, ABC):
    """PyTorch Dataset implementation for feature-target pairs.

    Wraps numpy arrays as PyTorch tensors for use with DataLoader.

    Args:
        x: Feature data as numpy array.
        y: Target data as numpy array.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Initialize dataset with feature and target arrays."""
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Get a single sample by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, target) tensors for the given index.
        """
        return self.x[idx], self.y[idx]


class TextDatasetHelperMixin:
    """Mixin providing helper methods for text processing and tokenization."""

    def tokenize_texts(self, texts: np.ndarray, spacy_model: str = "en_core_web_sm") -> list[list[str]]:
        """Tokenize texts using spaCy, handling 1D and 2D arrays."""
        result = []
        if type(texts) is np.ndarray and texts.ndim > 1:
            for i in range(texts.shape[1]):
                result.append(self.tokenize_texts(texts[:, i], spacy_model=spacy_model))
        elif type(texts) is np.ndarray and texts.ndim == 1:
            result.append(self._tokenize(texts, spacy_model=spacy_model))
        else:
            raise ValueError("Input texts must be a 1D or 2D numpy array.")
        return result

    def _tokenize(self, texts: np.ndarray, spacy_model: str) -> list[list[str]]:
        """Tokenize texts using spaCy.

        Args:
            texts: Iterable of text strings to tokenize.
            spacy_model: Name of the spaCy model to use (default: "en_core_web_sm").

        Returns:
            List of tokenized texts, where each text is a list of token strings.
        """
        nlp = spacy.load(spacy_model)
        tokenized_texts = []

        # Convert to list of Python strings, handling NaN values
        text_list = []
        for text in texts:
            if pd.isna(text) or text is np.nan:
                text_list.append("")
            else:
                text_list.append(str(text))

        for doc in nlp.pipe(text_list, disable=["parser", "ner"]):
            tokens = []
            if doc is not np.nan:
                tokens = [token.text for token in doc]
            tokenized_texts.append(tokens)
        return tokenized_texts

    def _flatten_text_lists(self, texts: list) -> list:
        result = []
        for element in texts:
            if isinstance(element, str):
                result.append(element)
            else:
                result.extend(self._flatten_text_lists(element))
        return result

    def build_vocabulary(
        self,
        texts: list,
        vocab_size: int,
        num_stopwords: int = 1,
    ) -> Vocabulary:
        """Build a vocabulary from tokenized texts.

        Creates a vocabulary with special tokens (<PAD>, <UNK>) and the most frequent tokens,
        optionally skipping the top N most common words (stopwords).

        Args:
            texts: List of tokenized texts (each text is a list of tokens).
            vocab_size: Maximum number of tokens to include in vocabulary (excluding special tokens).
            num_stopwords: Number of most frequent words to skip (default: 1).

        Returns:
            Vocabulary object with token-to-ID and ID-to-token mappings.
        """

        _texts = self._flatten_text_lists(texts)
        # count frequency of tokens
        counts = Counter(_texts)

        # configurable num_stopwords and vocab_size
        vocab = [x[0] for x in counts.most_common(vocab_size + num_stopwords)]
        special_tokens = ["<PAD>", "<UNK>"]  # Padding and unknown tokens
        tokentries = special_tokens + [*list(vocab[num_stopwords:])]
        id2tok = dict(zip(range(len(tokentries)), tokentries))

        # compute reverse index
        tok2id = dict(zip(tokentries, range(len(tokentries))))

        return Vocabulary(id2token=id2tok, token2id=tok2id)

    def pad_sequences(self, texts: list[list[str]]) -> list[list[str]]:
        """Pad sequences to the same length using <PAD> tokens.

        Args:
            texts: List of tokenized texts with varying lengths.

        Returns:
            List of padded sequences, all with the same length.
        """
        max_length = max(len(seq) for seq in texts)
        padded_sequences = []
        for seq in texts:
            padded = seq + ["<PAD>"] * (max_length - len(seq))
            padded_sequences.append(padded)
        return padded_sequences

    def tokens_to_indices(self, tokens: list[str], word_to_idx: dict[str, int]) -> list[int]:
        """Convert tokens to their vocabulary indices.

        Args:
            tokens: List of token strings.
            word_to_idx: Dictionary mapping tokens to their integer IDs.

        Returns:
            List of integer indices, using <UNK> index for unknown tokens.
        """
        return [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]


class TextBootstrapMixin(TextDatasetHelperMixin):
    """Mixin for bootstrapping text data with vocabulary creation."""

    def _get_vocabulary(self, x: np.ndarray, size: int) -> Vocabulary:
        """Build vocabulary from all sequences in the data.

        Args:
            x: 2D array where each column contains text sequences.
            size: Maximum vocabulary size.

        Returns:
            Vocabulary object built from all text sequences.
        """
        # sequences = []
        # for i in range(x.shape[1]):
        #     sequences.extend(x[:, i].tolist())

        tokenized_sequences = self.tokenize_texts(x)
        return self.build_vocabulary(tokenized_sequences, vocab_size=size)

    def bootstrap(self, x: np.ndarray, y: np.ndarray, vocabulary_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Process text data: tokenize, build vocabulary, pad, and convert to indices.

        Args:
            x: 2D array where each column contains text sequences.
            y: Target array.
            vocabulary_size: Maximum vocabulary size to build.

        Returns:
            Tuple of (processed_features, targets) where features are integer-encoded sequences.
        """
        self.vocabulary = self._get_vocabulary(x, size=vocabulary_size)
        all_sequences = []
        for i in range(x.shape[1]):
            sequences = x[:, i]
            tokenized_seqs = self.tokenize_texts(sequences.tolist())
            padded_seqs = self.pad_sequences(tokenized_seqs)
            indexed_sequences = [self.tokens_to_indices(seq, self.vocabulary.token2id) for seq in padded_seqs]
            all_sequences.append(indexed_sequences)

        x = np.column_stack(all_sequences)
        return x, y


class CsvLoader(Loader):
    """Loader for CSV files.

    Args:
        data_url: Path or URL to the CSV file.
        x_cols: List of column names to use as features.
        y_cols: List of column names to use as targets.
    """

    def __init__(self, data_url: str, x_cols: list[str], y_cols: list[str]):
        """Initialize the CSV loader with data source and column specifications."""
        self._data_url = data_url
        self._x_cols = x_cols
        self._y_cols = y_cols

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file.

        Fills NaN values in feature columns with empty strings.

        Returns:
            Tuple of (features, targets) as numpy arrays.
        """
        df = pd.read_csv(self._data_url)
        for col in self._x_cols:
            df.fillna({col: ""}, inplace=True)
        x = df[self._x_cols].values
        y = df[self._y_cols].values
        return x, y


class DataBootstrapper(SplitterMixin):
    """Bootstrap data loading and splitting for PyTorch datasets.

    Loads data, splits into train/validation/test sets, and creates PyTorch Dataset objects.

    Args:
        loader: Loader instance to load the data.
        validation_split: Fraction of data for validation (default: 0.2).
        test_split: Fraction of data for test set (default: 0.0).

    Attributes:
        train_ds: Training dataset.
        validation_ds: Validation dataset.
        test_ds: Test dataset (None if test_split is 0).
    """

    def __init__(self, loader: Loader, validation_split: float = 0.2, test_split: float = 0.0):
        """Initialize and bootstrap the datasets."""
        x, y = loader.load()
        train_data, validation_data, test_data = self.split(x, y, validation_split, test_split)
        self.train_ds = BaseDataset(train_data[0], train_data[1])
        self.validation_ds = BaseDataset(validation_data[0], validation_data[1])
        if test_data is not None:
            self.test_ds = BaseDataset(test_data[0], test_data[1])
        else:
            self.test_ds = None

    def get_dataloaders(self, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        """Create PyTorch DataLoaders for train, validation, and test sets.

        Args:
            batch_size: Number of samples per batch (default: 32).

        Returns:
            Tuple of (train_loader, validation_loader, test_loader).
            test_loader is None if no test set was created.
        """
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_ds, batch_size=batch_size, shuffle=False)
        if self.test_ds is not None:
            test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None
        return train_loader, validation_loader, test_loader


class TextDataBootstrapper(TextBootstrapMixin, SplitterMixin):
    """Bootstrap text data with tokenization, vocabulary building, and dataset creation.

    Extends DataBootstrapper with text-specific processing including tokenization,
    vocabulary creation, padding, and index conversion.

    Args:
        loader: Loader instance to load the text data.
        validation_split: Fraction of data for validation (default: 0.2).
        test_split: Fraction of data for test set (default: 0.0).
        vocabulary_size: Maximum vocabulary size (default: 10000).

    Attributes:
        train_ds: Training dataset with processed text.
        validation_ds: Validation dataset with processed text.
        test_ds: Test dataset with processed text (None if test_split is 0).
        vocabulary: Vocabulary object created from the training data.
    """

    def __init__(
        self, loader: Loader, validation_split: float = 0.2, test_split: float = 0.0, vocabulary_size: int = 10000
    ):
        """Initialize and bootstrap text datasets with vocabulary."""
        x, y = self.bootstrap(*loader.load(), vocabulary_size=vocabulary_size)
        train_data, validation_data, test_data = self.split(x, y, validation_split, test_split)
        self.train_ds = BaseDataset(train_data[0], train_data[1])
        self.validation_ds = BaseDataset(validation_data[0], validation_data[1])
        if test_data is not None:
            self.test_ds = BaseDataset(test_data[0], test_data[1])
        else:
            self.test_ds = None
