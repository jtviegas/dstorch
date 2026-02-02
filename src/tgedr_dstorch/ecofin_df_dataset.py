"""Dataset utilities for wrapping numpy arrays as PyTorch datasets and loaders."""

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EcoFinDataframeDataset(Dataset):
    """PyTorch Dataset for EcoFin dataframe data with tokenization."""

    @staticmethod
    def calculate_volatility(prices: np.ndarray | torch.Tensor, window: int = 5) -> np.ndarray:
        """Calculate rolling volatility from price series.

        Args:
            prices: Price series as numpy array or torch tensor
            window: Rolling window size for volatility calculation

        Returns:
            Numpy array of annualized volatility values (first window values are NaN)

        """
        # Convert tensor to numpy if needed
        if isinstance(prices, torch.Tensor):
            prices = prices.detach().cpu().numpy()
        
        prices = np.asarray(prices, dtype=np.float64)
        
        # Calculate log returns: ln(Pt / Pt-1)
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Calculate rolling standard deviation
        n = len(log_returns)
        volatility = np.full(len(prices), np.nan)
        
        for i in range(window - 1, n):
            # Get window of returns
            window_returns = log_returns[i - window + 1:i + 1]
            # Calculate std and annualize (252 trading days)
            volatility[i + 1] = np.std(window_returns, ddof=1) * np.sqrt(252)
        
        return volatility

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tokenizer: str = "yiyanghkust/finbert-tone",
        sequence_max_length: int = 512,
        volatility_window: int = 5,
    ) -> None:
        """Initialize the EcoFinDataframeDataset.

        Parameters
        ----------
        x : np.ndarray
            Input features array.
        y : np.ndarray
            Price series for volatility calculation.
        tokenizer : str, optional
            Name of the pretrained tokenizer to use.
        sequence_max_length : int, optional
            Maximum sequence length for tokenization.
        volatility_window : int, optional
            Window size for volatility calculation.

        """
        self._x = x
        self._y_prices = y
        # Calculate volatility from prices
        self._y = self.calculate_volatility(y, window=volatility_window)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._sequence_max_length = sequence_max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tokenized sample and its target label at the given index."""
        text = self._x[idx]

        encoding = self._tokenizer(
            text=str(text[0]),  # Sequence A: The Identity (ticker)
            text_pair=str(text[1]),  # Sequence B: The Context (news headline)
            add_special_tokens=True,
            max_length=self._sequence_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Target Volatility
        target = torch.tensor(self._y[idx], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": target,
        }
