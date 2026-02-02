"""Dataset utilities for wrapping numpy arrays as PyTorch datasets and loaders."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    """PyTorch Dataset implementation for feature-target pairs.

    Wraps numpy arrays as PyTorch tensors for use with DataLoader.

    Args:
        x: Feature data as numpy array.
        y: Target data as numpy array.

    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Initialize dataset with feature and target arrays."""
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, target) tensors for the given index.

        """
        return self.x[idx], self.y[idx]

    def to_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:  # noqa: FBT001, FBT002
        """Convert the dataset to a PyTorch DataLoader.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data at every epoch.

        Returns:
            DataLoader instance for the dataset.

        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
