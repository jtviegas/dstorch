from abc import ABC, abstractmethod
import subprocess
import sys
import numpy as np
import spacy
from spacy.cli import download as spacy_download


class Tokenizer(ABC):
    """Abstract base class for tokenizers.

    Subclasses must implement the tokenize method.
    """

    __ALLOWED_MODELS = ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf")
    _EMPTY_VAL = None

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
            import importlib
            try:
                importlib.import_module(self._model)
                self._np = spacy.load(self._model)
            except ImportError:
                subprocess.run([sys.executable, "-m", "spacy", "download", self._model], check=True)
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

        result = self.process_multidimensional(x=x.tolist())
        return np.array(result, dtype=object)

    def process_multidimensional(self, x: list) -> list:

        result = []
        if 1 == np.array(x).ndim:  # noqa: SIM300
            row = self.process_unidimensional(texts=x)
            result = row
        else:
            for s in x:
                row = self.process_multidimensional(x=s)
                result.append(row)
        return result

    def process_unidimensional(self, texts: list) -> list:
        result = []
        for t in texts:
            tokens = self.tokenize_string(s=str(t))
            result.append(tokens)
        return result

    def tokenize_string(self, s: str) -> list[str]:
        doc = self._np(s)
        tokens = [token.text for token in doc]
        if len(tokens) > self._sequence_max_len:
            tokens = tokens[: self._sequence_max_len]
        elif len(tokens) < self._sequence_max_len:
            tokens = tokens + [self._EMPTY_VAL] * (self._sequence_max_len - len(tokens))

        return tokens
