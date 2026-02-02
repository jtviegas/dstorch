"""Dataset utilities for wrapping numpy arrays as PyTorch datasets and loaders."""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EcoFinDataframeDataset(Dataset):
    """PyTorch Dataset for EcoFin dataframe data with tokenization."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tokenizer: str = "yiyanghkust/finbert-tone",
        sequence_max_length: int = 512,
    ) -> None:
        """Initialize the EcoFinDataframeDataset.

        Parameters
        ----------
        x : np.ndarray
            Input features array.
        y : np.ndarray
            Target values array.
        tokenizer : str, optional
            Name of the pretrained tokenizer to use.
        sequence_max_length : int, optional
            Maximum sequence length for tokenization.

        """
        self._x = x
        self._y = y
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._sequence_max_length = sequence_max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tokenized sample and its target label at the given index."""
        text = self._x[idx]

        encoding = self._tokenizer._encode_plus(
            text=text[0].item(),  # Sequence A: The Identity
            text_pair=text[1].item(),  # Sequence B: The Context
            add_special_tokens=True,
            max_length=self._sequence_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 2. Target Volatility
        target = torch.tensor(self._y[idx], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": target,
        }
