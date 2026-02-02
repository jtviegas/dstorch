"""Tokenizer abstractions and implementations based on spaCy.

This module provides:
- Tokenizer: abstract base class for tokenizer implementations.
- BaseTokenizer: concrete tokenizer for numpy array inputs.
"""

from abc import ABC, abstractmethod
import subprocess
import sys
import numpy as np
import spacy


class Tokenizer(ABC):
    """Abstract base class for tokenizers.

    Subclasses must implement the tokenize method.
    """

    __ALLOWED_MODELS = ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf")

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """Initialize the loader with optional configuration.

        Args:
            model: Spacy model name to use for tokenization.

        """
        self._model = model
        if model not in self.__ALLOWED_MODELS:
            msg = f"model must be one of {self.__ALLOWED_MODELS}."
            raise ValueError(msg)

        try:
            self._np = spacy.load(self._model)
        except OSError:
            import importlib
            try:
                importlib.import_module(self._model)
                self._np = spacy.load(self._model)
            except ImportError:
                subprocess.run([sys.executable, "-m", "spacy", "download", self._model], check=True)  # noqa: S603
                self._np = spacy.load(self._model)

    def get_default() -> "Tokenizer":
        """Get a default Tokenizer instance.

        Returns:
            Tokenizer instance with default configuration.

        """
        return BaseTokenizer()


    @abstractmethod
    def tokenize(self, x: np.ndarray, sequence_len: int = 30000, padding: str = "<PAD>") -> np.ndarray:
        """Tokenize a list of texts.

        Args:
            x: Numpy array of texts to tokenize.
            sequence_len: Maximum length of the tokenized sequence.
            padding: Token to use for padding shorter sequences.

        Returns:
            Numpy array of tokenized texts.

        """
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)


class BaseTokenizer(Tokenizer):
    """Concrete tokenizer implementation for processing numpy arrays of texts."""

    def tokenize(self, x: np.ndarray, sequence_len: int = 30000, padding: str = "<PAD>") -> np.ndarray:
        """Tokenize a list of texts.

        Args:
            x: List of strings to tokenize.
            sequence_len: Maximum length of the tokenized sequence.
            padding: Token to use for padding shorter sequences.

        Returns:
            Numpy array of tokenized texts.

        """
        if 2 != len(x.shape):  # noqa: SIM300
            msg = "x must be a 2D numpy array."
            raise ValueError(msg)

        result = self.process_multidimensional(x=x.tolist(), sequence_len=sequence_len, padding=padding)
        return np.array(result, dtype=object)

    def process_multidimensional(self, x: list, sequence_len: int = 30000, padding: str = "<PAD>") -> list:
        """Recursively tokenize nested lists of texts.

        Args:
            x: Nested list of texts to tokenize.
            sequence_len: Maximum length of the tokenized sequence.
            padding: Token to use for padding shorter sequences.

        Returns:
            Nested list of tokenized texts.

        """
        result = []
        if 1 == np.array(x).ndim:  # noqa: SIM300
            row = self.process_unidimensional(texts=x)
            result = row
        else:
            for s in x:
                row = self.process_multidimensional(x=s, sequence_len=sequence_len, padding=padding)
                result.append(row)
        return result

    def process_unidimensional(self, texts: list, sequence_len: int = 30000, padding: str = "<PAD>") -> list:
        """Tokenize a list of text entries into fixed-length token sequences.

        Args:
            texts: List of text entries to tokenize.
            sequence_len: Maximum length of the tokenized sequence.
            padding: Token to use for padding shorter sequences.

        Returns:
            List of tokenized text sequences.

        """
        result = []
        for t in texts:
            tokens = self.tokenize_string(s=str(t), sequence_len=sequence_len, padding=padding)
            result.append(tokens)
        return result

    def tokenize_string(self, s: str, sequence_len: int = 30000, padding: str = "<PAD>") -> list[str]:
        """Tokenize a single string into a fixed-length list of tokens.

        Args:
            s: Input string to tokenize.
            sequence_len: Maximum length of the tokenized sequence.
            padding: Token to use for padding shorter sequences.

        Returns:
            List of tokens padded or truncated to sequence_len.

        """
        doc = self._np(s)
        tokens = [token.text for token in doc]
        if len(tokens) > sequence_len:
            tokens = tokens[: sequence_len]
        elif len(tokens) < sequence_len:
            tokens = tokens + [padding] * (sequence_len - len(tokens))

        return tokens
