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
from tgedr_dstorch.loader import CsvLoader, Loader



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
