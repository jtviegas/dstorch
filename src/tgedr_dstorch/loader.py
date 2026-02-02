from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Loader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement the load method to load data from various sources.
    """

    _NA_REPLACEMENT_KEY = "na_replacement"
    _DEFAULT_NA_REPLACEMENT = ""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the loader with optional configuration.

        Args:
            config: Optional configuration dictionary for the loader.

        """
        self._config = config
        self._na_replacement = (self._DEFAULT_NA_REPLACEMENT
                                if (config is None or self._NA_REPLACEMENT_KEY not in config)
                                else config[self._NA_REPLACEMENT_KEY]
                                )


    @abstractmethod
    def load(self, url: str, x_cols: list[str] | None = None, y_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Load and return feature and target data.

        Args:
            url: str, the URL or path to the data source.
            x_cols: List of column names for features. If None, defaults to all columns except the last.
            y_cols: List of column names for targets. If None, defaults to the last column.

        Returns:
            Tuple of (features, targets) as numpy arrays.

        """
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_default() -> "Loader":
        """Get a default Loader instance.

        Returns:
            Loader instance with default configuration.

        """
        return CsvLoader()


class CsvLoader(Loader):
    """Loader for CSV files."""

    def load(self, url: str, x_cols: list[str] | None = None, y_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file.

        Args:
            url: str, the URL or path to the data source.
            x_cols: List of column names for features. If None, defaults to all columns except the last.
            y_cols: List of column names for targets. If None, defaults to the last column.

        Returns:
            Tuple of (features, targets) as numpy arrays.

        """
        df = pd.read_csv(url)  # noqa: PD901
        x_cols = x_cols if x_cols is not None else df.columns[:-1].tolist()
        y_cols = y_cols if y_cols is not None else [df.columns[-1]]

        for col in x_cols:
            df = df.fillna({col: self._na_replacement})  # noqa: PD901
        x = df[x_cols].to_numpy()
        y = df[y_cols].to_numpy()
        return x, y
