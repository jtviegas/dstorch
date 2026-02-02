import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import torch

from tgedr_dstorch.dataset import BaseDataset
from tgedr_dstorch.text_dataset import TextDatasetBuilder


def test_base_dataset_simple_array():
    """Test BaseDataset with a simple 3x2 array containing integer columns."""
    # Create a simple 3x2 array with integers
    x = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=np.int32)
    
    # Targets as a 3x1 array
    y = np.array([[0], [1], [0]], dtype=np.int32)
    
    # Create dataset
    dataset = BaseDataset(x=x, y=y)
    
    # Test length
    assert len(dataset) == 3
    
    # Test __getitem__
    x0, y0 = dataset[0]
    assert x0.tolist() == [1, 2]
    assert y0.tolist() == [0]
    
    x1, y1 = dataset[1]
    assert x1.tolist() == [3, 4]
    assert y1.tolist() == [1]
    
    x2, y2 = dataset[2]
    assert x2.tolist() == [5, 6]
    assert y2.tolist() == [0]
    
    # Test that tensors have correct dtype
    assert x0.dtype.is_floating_point == False
    assert y0.dtype.is_floating_point == False


def test_base_dataset_to_dataloader():
    """Test converting BaseDataset to DataLoader."""
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    y = np.array([[0], [1], [0]], dtype=np.int32)
    
    dataset = BaseDataset(x=x, y=y)
    dataloader = dataset.to_dataloader(batch_size=2, shuffle=False)
    
    # Test dataloader
    assert dataloader.batch_size == 2
    assert len(dataloader) == 2  # 3 samples / 2 batch_size = 2 batches (rounded up)
    
    # Iterate and check first batch
    for batch_x, batch_y in dataloader:
        assert batch_x.shape[0] <= 2  # At most 2 samples per batch
        assert batch_y.shape[0] <= 2
        break


@pytest.fixture
def simple_text_dataset_csv(tmp_path):
    """Create a simple CSV with 6 rows, 2 text columns, and 1 integer target column."""
    csv_path = tmp_path / "simple_text_data.csv"
    
    # Create a simple dataset with 6 rows
    data = {
        "text_col1": ["hello world", "foo bar", "test data", "sample text", "another row", "final entry"],
        "text_col2": ["first line", "second line", "third line", "fourth line", "fifth line", "sixth line"],
        "label": [0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


def test_text_dataset_builder_simple(simple_text_dataset_csv):
    """Test TextDatasetBuilder with a simple 6-row dataset with 25% validation and 25% test split."""
    # Create the builder with 25% validation and 25% test
    builder = TextDatasetBuilder(
        x_cols=["text_col1", "text_col2"],
        y_cols=["label"],
        sequence_len=100,
        vocabulary_size=50,
        validation_split=0.25,
        test_split=0.25
    )
    
    # Build the datasets
    builder.build(url=simple_text_dataset_csv)
    
    # Verify vocabulary was built
    assert builder.vocabulary is not None
    
    # Verify all datasets were created
    assert builder.train is not None
    assert builder.validation is not None
    assert builder.test is not None
    
    # Verify split sizes (6 rows total: 50% train = 3, 25% val = 1-2, 25% test = 1-2)
    train_len = len(builder.train)
    val_len = len(builder.validation)
    test_len = len(builder.test)
    
    # Total should be 6
    assert train_len + val_len + test_len == 6
    
    # Train should be approximately 50% (3 samples)
    assert 2 <= train_len <= 4  # Allow some variance due to rounding
    
    # Validation and test should each be approximately 25%
    assert val_len >= 1
    assert test_len >= 1
    
    # Verify that datasets return tensors
    x_train, y_train = builder.train[0]
    assert x_train is not None
    assert y_train is not None

def test_text_dataset_builder_simple_no_test_split(simple_text_dataset_csv):
    """Test TextDatasetBuilder with a simple 6-row dataset with 25% validation and no test split."""
    # Create the builder with 25% validation and no test split
    builder = TextDatasetBuilder(
        x_cols=["text_col1", "text_col2"],
        y_cols=["label"],
        sequence_len=100,
        vocabulary_size=50,
        validation_split=0.5
    )
    
    # Build the datasets
    builder.build(url=simple_text_dataset_csv)
    
    # Verify vocabulary was built
    assert builder.vocabulary is not None
    
    # Verify all datasets were created
    assert builder.train is not None
    assert builder.validation is not None
    assert builder.test is None
    
    # Verify split sizes (6 rows total: 50% train = 3, 25% val = 1-2, 25% test = 1-2)
    train_len = len(builder.train)
    val_len = len(builder.validation)
    
    # Total should be 6
    assert train_len + val_len == 6
    
    # Train should be approximately 50% (3 samples)
    assert 2 <= train_len <= 4  # Allow some variance due to rounding
    
    # Validation and test should each be approximately 25%
    assert val_len >= 1
    
    # Verify that datasets return tensors
    x_train, y_train = builder.train[0]
    assert x_train is not None
    assert x_train.dtype == torch.int64
    assert x_train.dim() == 2
    assert y_train is not None
    assert y_train.dtype == torch.int64
    assert y_train.dim() == 1