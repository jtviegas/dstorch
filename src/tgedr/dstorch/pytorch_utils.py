from abc import abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


@dataclass(frozen=True)
class Datasets:
    """Container for train, validation, and test datasets.

    Attributes:
        train: Training dataset
        validation: Optional validation dataset
        test: Optional test dataset

    """

    train: Dataset
    validation: Dataset | None
    test: Dataset | None


class DatasetsBuilder:
    """Abstract base class for building train/validation/test dataset splits.

    This class handles the split ratios and provides a template for loading
    and building datasets from a data source.

    Attributes:
        x: Feature data as numpy array
        y: Label data as numpy array
        train_ratio: Proportion of data used for training
        validation_split: Proportion of data used for validation
        test_split: Proportion of data used for testing

    """

    def __init__(self, data_url, validation_split=0.2, test_split=0.0):
        """Initialize the datasets builder.

        Args:
            data_url: Path or URL to the data source
            validation_split: Proportion for validation set (default: 0.2)
            test_split: Proportion for test set (default: 0.0)

        Raises:
            AssertionError: If ratios don't sum to 1.0

        """
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.train_ratio = 1.0 - validation_split - test_split
        assert abs(self.train_ratio + validation_split + test_split - 1.0) < 1e-6, "Ratios must sum to 1"
        self.validation_split = validation_split
        self.test_split = test_split
        self._load_dataset(data_url)

    @abstractmethod
    def _load_dataset(self, data_url: str):
        """Load dataset from the given URL.

        Args:
            data_url: Path or URL to the data source

        Raises:
            NotImplementedError: Must be implemented by subclasses

        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    @abstractmethod
    def build_datasets(self, data_url: str, validation_split=0.2, test_split=0.0) -> Datasets:
        """Build train/validation/test datasets from the data source.

        Args:
            data_url: Path or URL to the data source
            validation_split: Proportion for validation set (default: 0.2)
            test_split: Proportion for test set (default: 0.0)

        Returns:
            Datasets container with train, validation, and test splits

        Raises:
            NotImplementedError: Must be implemented by subclasses

        """
        raise NotImplementedError("This method must be implemented by subclasses.")


class TextLabelDataset(Dataset):
    """PyTorch Dataset for text and label pairs with train/validation/test splits.

    This abstract class extends PyTorch's Dataset and provides functionality to
    load text-label data and split it into train, validation, and test sets.

    Attributes:
        texts: List of text samples
        labels: List of corresponding labels
        train_ratio: Proportion of data used for training
        validation_split: Proportion of data used for validation
        test_split: Proportion of data used for testing
        train_data: Tuple of training texts and labels as tensors
        val_data: Tuple of validation texts and labels as tensors
        test_data: Optional tuple of test texts and labels as tensors

    """

    def __init__(self, data_url, validation_split=0.2, test_split=0.0):
        """Initialize the text-label dataset.

        Args:
            data_url: Path or URL to the data source
            validation_split: Proportion for validation set (default: 0.2)
            test_split: Proportion for test set (default: 0.0)

        Raises:
            AssertionError: If ratios don't sum to 1.0

        """
        self.texts: list[str] = None
        self.labels: list[float] = None
        self.train_ratio = 1.0 - validation_split - test_split
        assert abs(self.train_ratio + validation_split + test_split - 1.0) < 1e-6, "Ratios must sum to 1"
        self.validation_split = validation_split
        self.test_split = test_split
        self._load_dataset(data_url)
        self._set_splits()

    @abstractmethod
    def _load_dataset(self, data_url: str):
        """Load dataset from the given URL.

        Args:
            data_url: Path or URL to the data source

        Raises:
            NotImplementedError: Must be implemented by subclasses

        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _set_splits(self):
        """Split the loaded data into train, validation, and test sets.

        Creates train_data, val_data, and test_data attributes containing
        tuples of (texts, labels) as PyTorch tensors.

        """
        texts_train, texts_temp, labels_train, labels_temp = train_test_split(
            self.texts, self.labels, test_size=(self.validation_split + self.test_split), random_state=42
        )
        if 1e-6 > self.test_split:
            self.train_data = (torch.Tensor(texts_train), torch.FloatTensor(labels_train))
            self.val_data = (torch.Tensor(texts_temp), torch.FloatTensor(labels_temp))
            self.test_data = None
        else:
            val_size = self.validation_split / (self.validation_split + self.test_split)
            texts_val, texts_test, labels_val, labels_test = train_test_split(
                texts_temp, labels_temp, test_size=(1 - val_size), random_state=42
            )
            self.train_data = (torch.Tensor(texts_train), torch.FloatTensor(labels_train))
            self.val_data = (torch.Tensor(texts_val), torch.FloatTensor(labels_val))
            self.test_data = (torch.Tensor(texts_test), torch.FloatTensor(labels_test))

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            Number of text-label pairs

        """
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a single text-label pair by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (text, label) at the given index

        """
        return self.texts[idx], self.labels[idx]


class SentimentAnalysisNN(nn.Module):
    """Neural network model for sentiment analysis using LSTM.

    This model uses an embedding layer, bidirectional LSTM, and a linear output layer
    for sentiment classification tasks.

    Attributes:
        embedding_layer: Embedding layer for input tokens
        _dim_input: Input vocabulary dimension
        _dim_embedding: Embedding dimension
        _dim_hidden: Hidden layer dimension
        _dim_output: Output dimension
        _loss_func: Loss function for training
        _optimizer: Optimizer for training
        _hidden_layers: List of hidden LSTM layers
        _output_layer: Final linear output layer

    """

    def __init__(
        self, input_dim, embedding_dim, hidden_dim, output_dim, embedding_layer=None, loss_func=None, optimizer=None
    ):
        """Initialize the sentiment analysis neural network.

        Args:
            input_dim: Vocabulary size for embeddings
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of LSTM hidden state
            output_dim: Dimension of output layer
            embedding_layer: Optional pre-trained embedding layer (default: None)
            loss_func: Optional loss function (default: BCEWithLogitsLoss)
            optimizer: Optional optimizer (default: Adam)

        """
        super().__init__()
        self._dim_input = input_dim
        self._dim_embedding = embedding_dim
        self._dim_hidden = hidden_dim
        self._dim_output = output_dim
        self.embedding_layer = (
            nn.Embedding(self._dim_input, self._dim_embedding) if embedding_layer is None else embedding_layer
        )
        self._loss_func = nn.BCEWithLogitsLoss() if loss_func is None else loss_func
        self._optimizer = torch.optim.Adam(self.parameters()) if optimizer is None else optimizer
        self._hidden_layers = []
        self._output_layer = None
        self.build_model()

    def build_model(self):
        """Build the model architecture.

        Creates a bidirectional LSTM layer and a linear output layer.

        """
        self._hidden_layers.append(nn.LSTM(self._dim_embedding, self._dim_hidden, bidirectional=True))
        self._output_layer = nn.Linear(self._dim_hidden * 2, self._dim_output)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of token indices

        Returns:
            Predictions tensor

        """
        embedded = self.embedding_layer(x)
        output, _ = self.rnn(embedded)
        avg_pool = torch.mean(output, dim=0)
        predictions = self.fc(avg_pool)
        return predictions

    def batch_train(self, epochs: int = 7, device=None) -> float:
        """Train the model using batch processing.

        Args:
            epochs: Number of training epochs (default: 7)
            device: Device to train on (CPU or CUDA), optional

        Returns:
            Final average loss from the last epoch

        """
        if device is not None:
            self.to(device)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch in train_iterator:
                text, labels = batch.text, batch.label
                optimizer.zero_grad()
                predictions = model(text).squeeze(1)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            average_loss = running_loss / len(train_iterator)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
