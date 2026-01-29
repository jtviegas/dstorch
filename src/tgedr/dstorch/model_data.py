"""Data utilities for machine learning model datasets.

This module provides:
- ModelData: Abstract base class for ML model datasets.
"""

from abc import ABC, abstractmethod
import subprocess
import sys
import numpy as np
import pandas as pd
import spacy
from tgedr.dstorch.loader import CsvLoader, Loader


class Tokenizer(ABC):
    """Abstract base class for tokenizers.

    Subclasses must implement the tokenize method.
    """

    __ALLOWED_MODELS = ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf")

    def __init__(self, sequence_max_len: int = 30000, model: str = "en_core_web_sm") -> None:
        """Initialize the loader with optional configuration.

        Args:
            sequence_max_len: Maximum length of the tokenized sequence.
            model: Spacy model name to use for tokenization.

        """
        self._sequence_max_len = sequence_max_len
        self._model = model
        if model not in self.__ALLOWED_MODELS:
            msg = f"model must be one of {self.__ALLOWED_MODELS}."
            raise ValueError(msg)
        try:
            self._np = spacy.load(self._model)
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", self._model], check=True)  # noqa: S603
            self._np = spacy.load(self._model)

    @abstractmethod
    def tokenize(self, x: np.ndarray) -> np.ndarray:
        """Tokenize a list of texts.

        Args:
            texts: Numpy array of texts to tokenize.

        Returns:
            Numpy array of tokenized texts.

        """
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

class BaseTokenizer(Tokenizer):

    def tokenize(self, x: np.ndarray) -> np.ndarray:
        """Tokenize a list of texts.

        Args:
            x: List of strings to tokenize.

        Returns:
            Numpy array of tokenized texts.

        """
        if 2 != len(x.shape):  # noqa: SIM300
            msg = "x must be a 2D numpy array."
            raise ValueError(msg)

        result: np.ndarray = np.full((x.shape[0], x.shape[1]), np.nan)

        for i in range(x.shape[1]):
            text = x[i]
            tokenized_textset: np.ndarray = np.full((x.shape[0], self._sequence_max_len), np.nan)

            for doc in self._np.pipe(text, disable=["parser", "ner"]):
                tokens = []
                if doc is not np.nan:
                    tokens = np.array([token.text for token in doc])
                    #tokenized_textset.append(tokens)
                #tokenized_texts.append(tokens)

        return result #np.array(tokenized_texts)




class ModelData:
    """Abstract base class for ML model datasets.

    Subclasses must implement the load method to load data from various sources.
    """

    @staticmethod
    def get_instance(data_url: str, loader: Loader | None = None) -> "ModelData":
        """Create a ModelData instance with optional loader configuration.

        Args:
            data_url: The URL or path to the data source.
            loader: Optional loader instance. Defaults to CsvLoader with predefined columns.

        Returns:
            ModelData instance initialized with the provided data URL and loader.

        """
        loader = loader if loader else CsvLoader()
        return ModelData(data_url, loader=loader)

    def __init__(self, data_url: str, loader: Loader) -> None:
        """Initialize the ModelData instance.

        Args:
            data_url: The URL or path to the data source.
            loader: The loader instance to use for loading data.

        """
        self._data_url = data_url
        self._loader: Loader = loader
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.bootstrap()

    def bootstrap(self) -> None:
        """Load and return feature and target data.

        Returns:
            Tuple of (features, targets) as numpy arrays.

        """
        self._x, self._y = self._loader.load(self._data_url)
