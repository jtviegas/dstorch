from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Loader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement the load method to load data from various sources.
    """

    def __init__(self, config: tuple[list[str], list[str]] | None = None) -> None:
        """Initialize the loader with optional configuration.

        Args:
            config: Optional configuration dictionary for the loader.

        """
        self._config = config

    @abstractmethod
    def load(self, url: str) -> tuple[np.ndarray, np.ndarray]:
        """Load and return feature and target data.

        Args:
            url: str, the URL or path to the data source.

        Returns:
            Tuple of (features, targets) as numpy arrays.

        """
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)


class CsvLoader(Loader):
    """Loader for CSV files."""

    def load(self, url: str) -> tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file.

        Args:
            url: str, the URL or path to the data source.

        Returns:
            Tuple of (features, targets) as numpy arrays.

        """
        if self._config and (len(self._config) != 2):
            msg = "config must be a tuple of two lists: (x_cols, y_cols)."
            raise ValueError(msg)

        df = pd.read_csv(url)  # noqa: PD901
        x_cols, y_cols = self._config if self._config else (df.columns[:-1].tolist(), [df.columns[-1]])

        for col in x_cols:
            df = df.fillna({col: ""})  # noqa: PD901
        x = df[x_cols].to_numpy()
        y = df[y_cols].to_numpy()
        return x, y
