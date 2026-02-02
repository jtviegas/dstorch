"""Dataset builders and PyTorch dataset utilities for text data."""

import numpy as np
from sklearn.model_selection import train_test_split
from tgedr_dstorch.dataset import BaseDataset
from tgedr_dstorch.loader import Loader
from tgedr_dstorch.tokenizer import Tokenizer
from tgedr_dstorch.vocabulary import Vocabulary
from tgedr_dstorch.vocabulary import VocabularyBuilder
from tgedr_pycommons.data.processing import process_text_array


class TextDatasetBuilder:
    """Build datasets for text classification with tokenization and splitting.

    Attributes:
        vocabulary: Built vocabulary for encoding text.
        train: Training dataset.
        validation: Validation dataset.
        test: Optional test dataset.

    """

    def __init__(
        self,
        loader_impl: Loader | None = None,
        tokenizer_impl: Tokenizer | None = None,
        x_cols: list[str] | None = None,
        y_cols: list[str] | None = None,
        sequence_len: int = 512,
        padding_token: str = "<PAD>",  # noqa: S107
        unknown_token: str = "<UNK>",  # noqa: S107
        vocabulary_size: int = 25000,
        num_stopwords: int = 0,
        validation_split: float = 0.2,
        test_split: float = 0.0,
    ) -> None:
        """Initialize the dataset builder configuration and dependencies."""
        self._loader = loader_impl if loader_impl else Loader.get_default()
        self._tokenizer = tokenizer_impl if tokenizer_impl else Tokenizer.get_default()
        self._xcols: list[str] = x_cols
        self._ycols: list[str] = y_cols
        self._sequence_len: int = sequence_len
        self._padding_token: str = padding_token
        self._unknown_token: str = unknown_token
        self._vocabulary_size: int = vocabulary_size
        self._num_stopwords: int = num_stopwords
        self._validation_split: float = validation_split
        self._test_split: float = test_split

        self.vocabulary: Vocabulary = None
        self.train: BaseDataset = None
        self.validation: BaseDataset = None
        self.test: BaseDataset = None


    def build(self, url: str) -> tuple[np.ndarray, np.ndarray]:
        """Build dataset from the given URL."""
        x, y = self._loader.load(url=url, x_cols=self._xcols, y_cols=self._ycols)
        x_tokenized = self._tokenizer.tokenize(x=x, sequence_len=self._sequence_len, padding=self._padding_token)
        self.vocabulary = VocabularyBuilder().build_vocabulary(
            texts=x_tokenized.tolist(),
            vocab_size=self._vocabulary_size,
            num_stopwords=self._num_stopwords,
            padding_token=self._padding_token,
            unknown_token=self._unknown_token,
        )
        x_encoded = process_text_array(x=x_tokenized.tolist(), f=self.vocabulary.encode)

        train_data, validation_data, test_data = self._split(
            x=np.array(x_encoded),
            y=np.array(y),
            validation_split=self._validation_split,
            test_split=self._test_split,
          )
        self.train = BaseDataset(x=train_data[0], y=train_data[1])
        self.validation = BaseDataset(x=validation_data[0], y=validation_data[1])
        if test_data is not None:
            self.test = BaseDataset(x=test_data[0], y=test_data[1])


    def _split(self, x: np.ndarray, y: np.ndarray, validation_split: float = 0.2, test_split: float = 0.0) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
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
        if 1e-6 > test_split:  # noqa: SIM300
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
