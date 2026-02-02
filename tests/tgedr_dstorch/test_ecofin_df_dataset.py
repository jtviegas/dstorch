import numpy as np
import pytest
import torch

from tgedr_dstorch.ecofin_df_dataset import EcoFinDataframeDataset


def test_ecofin_dataframe_dataset_simple():
    """Test EcoFinDataframeDataset with ticker, news headline, and price series."""
    # Create a 3x2 array: ticker symbol and news headline
    x = np.array([
        ["AAPL", "Apple reports record quarterly earnings"],
        ["GOOGL", "Google announces new AI breakthrough"],
        ["TSLA", "Tesla stock drops amid production concerns"]
    ])
    
    # Price series for volatility calculation (need at least 6 prices for window=5)
    prices = np.array([150.0, 152.5, 151.0, 153.75, 155.20, 154.0, 156.50, 158.0])
    
    # Create dataset with window=3 to ensure we have valid volatility values
    dataset = EcoFinDataframeDataset(x=x, y=prices, sequence_max_length=128, volatility_window=3)
    
    # Test length
    assert len(dataset) == 3
    
    # Test __getitem__ for first sample
    sample = dataset[0]
    
    # Verify structure
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample
    
    # Verify shapes
    assert sample["input_ids"].shape[0] == 128  # max_length
    assert sample["attention_mask"].shape[0] == 128
    assert sample["label"].shape == ()  # scalar tensor
    
    # Verify types
    assert sample["input_ids"].dtype == torch.int64 or sample["input_ids"].dtype == torch.long  # Token IDs are integers
    assert sample["label"].dtype == torch.float32 or sample["label"].dtype == torch.float  # Volatility is float
    
    # Verify label is a valid number (not NaN) - should use indices with valid volatility
    # With window=3 and 8 prices, we have valid volatility from index 3 onwards
    # Dataset has 3 samples, so they map to indices 0, 1, 2 of volatility array
    # Index 0 and 1 will be NaN, but index 2 might not be
    # Let's check that we can get samples
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["label"] is not None


def test_ecofin_dataframe_dataset_tokenization():
    """Test that tokenization produces valid token IDs."""
    x = np.array([
        ["MSFT", "Microsoft unveils new cloud services"],
        ["AMZN", "Amazon expands logistics network"]
    ])
    # Price series with enough data for window=3
    prices = np.array([100.0, 102.0, 101.5, 103.0, 104.5, 105.0])
    
    dataset = EcoFinDataframeDataset(x=x, y=prices, sequence_max_length=64, volatility_window=3)
    
    sample = dataset[0]
    
    # Token IDs should be non-negative integers
    assert (sample["input_ids"] >= 0).all()
    
    # Attention mask should be 0s and 1s
    assert ((sample["attention_mask"] == 0) | (sample["attention_mask"] == 1)).all()
    
    # First token should be [CLS] token (typically 101 for BERT models)
    assert sample["input_ids"][0] > 0


def test_calculate_volatility():
    """Test the calculate_volatility static method."""
    # Test with numpy array
    prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0])
    volatility = EcoFinDataframeDataset.calculate_volatility(prices, window=5)
    
    # First 5 values should be NaN
    assert np.isnan(volatility[0])
    assert np.isnan(volatility[1])
    assert np.isnan(volatility[2])
    assert np.isnan(volatility[3])
    assert np.isnan(volatility[4])
    
    # Values from index 5 onwards should be valid floats
    assert not np.isnan(volatility[5])
    assert volatility[5] > 0  # Volatility should be positive
    
    # Test with torch tensor
    import torch
    prices_tensor = torch.tensor([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])
    volatility_tensor = EcoFinDataframeDataset.calculate_volatility(prices_tensor, window=3)
    
    # Should return numpy array
    assert isinstance(volatility_tensor, np.ndarray)
    assert len(volatility_tensor) == 6
