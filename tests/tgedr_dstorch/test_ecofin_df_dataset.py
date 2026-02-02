import numpy as np
import pytest

from tgedr_dstorch.ecofin_df_dataset import EcoFinDataframeDataset


def test_ecofin_dataframe_dataset_simple():
    """Test EcoFinDataframeDataset with ticker, news headline, and float target."""
    # Create a 3x2 array: ticker symbol and news headline
    x = np.array([
        ["AAPL", "Apple reports record quarterly earnings"],
        ["GOOGL", "Google announces new AI breakthrough"],
        ["TSLA", "Tesla stock drops amid production concerns"]
    ])
    
    # Target values (volatility as floats)
    y = np.array([0.15, 0.42, 0.28])
    
    # Create dataset
    dataset = EcoFinDataframeDataset(x=x, y=y, sequence_max_length=128)
    
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
    assert sample["input_ids"].dtype == int  # Token IDs are integers
    assert sample["label"].dtype == float  # Volatility is float
    
    # Verify label value
    assert sample["label"].item() == 0.15
    
    # Test second sample
    sample2 = dataset[1]
    assert sample2["label"].item() == 0.42
    
    # Test third sample
    sample3 = dataset[2]
    assert sample3["label"].item() == 0.28


def test_ecofin_dataframe_dataset_tokenization():
    """Test that tokenization produces valid token IDs."""
    x = np.array([
        ["MSFT", "Microsoft unveils new cloud services"],
        ["AMZN", "Amazon expands logistics network"]
    ])
    y = np.array([0.5, 0.6])
    
    dataset = EcoFinDataframeDataset(x=x, y=y, sequence_max_length=64)
    
    sample = dataset[0]
    
    # Token IDs should be non-negative integers
    assert (sample["input_ids"] >= 0).all()
    
    # Attention mask should be 0s and 1s
    assert ((sample["attention_mask"] == 0) | (sample["attention_mask"] == 1)).all()
    
    # First token should be [CLS] token (typically 101 for BERT models)
    assert sample["input_ids"][0] > 0
