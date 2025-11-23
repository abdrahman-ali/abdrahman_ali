"""
PyTorch deep learning trainer.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


if PYTORCH_AVAILABLE:
    class FeedForwardNN(nn.Module):
        """Feedforward neural network."""

        def __init__(self, input_size: int, hidden_layers: list, output_size: int,
                    dropout: float = 0.2, batch_norm: bool = True):
            """
            Initialize feedforward network.

            Args:
                input_size: Input dimension
                hidden_layers: List of hidden layer sizes
                output_size: Output dimension
                dropout: Dropout rate
                batch_norm: Whether to use batch normalization
            """
            super(FeedForwardNN, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, output_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            """Forward pass."""
            return self.network(x)


    class PyTorchTrainer:
        """PyTorch model trainer."""

        def __init__(self, model: nn.Module, config: Optional[Dict] = None):
            """
            Initialize PyTorch trainer.

            Args:
                model: PyTorch model
                config: Training configuration
            """
            self.model = model
            self.config = config or {}
            self.device = self._get_device()
            self.model.to(self.device)

            # Training history
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }

        def _get_device(self) -> torch.device:
            """Get compute device."""
            device_config = self.config.get('device', 'auto')

            if device_config == 'auto':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    logger.info("Using CUDA")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device('mps')
                    logger.info("Using MPS (Apple Silicon)")
                else:
                    device = torch.device('cpu')
                    logger.info("Using CPU")
            else:
                device = torch.device(device_config)

            return device

        def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 epochs: int = 100, learning_rate: float = 0.001,
                 optimizer_name: str = 'adam', loss_fn: Optional[nn.Module] = None) -> Dict[str, list]:
            """
            Train the model.

            Args:
                train_loader: Training data loader
                val_loader: Validation data loader
                epochs: Number of epochs
                learning_rate: Learning rate
                optimizer_name: Optimizer name
                loss_fn: Loss function

            Returns:
                Training history
            """
            logger.info(f"Starting training for {epochs} epochs...")

            # Setup optimizer
            optimizer = self._get_optimizer(optimizer_name, learning_rate)

            # Setup loss function
            if loss_fn is None:
                loss_fn = nn.CrossEntropyLoss()

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.config.get('early_stopping', {}).get('patience', 10)

            for epoch in range(epochs):
                # Train
                train_loss, train_acc = self._train_epoch(train_loader, optimizer, loss_fn)
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

                # Validate
                if val_loader is not None:
                    val_loss, val_acc = self._validate_epoch(val_loader, loss_fn)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_checkpoint('best_model.pt')
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            logger.info("Training completed!")
            return self.history

        def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                        loss_fn: nn.Module) -> Tuple[float, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total

            return avg_loss, accuracy

        def _validate_epoch(self, val_loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
            """Validate for one epoch."""
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = loss_fn(outputs, batch_y)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_loss = total_loss / len(val_loader)
            accuracy = correct / total

            return avg_loss, accuracy

        def _get_optimizer(self, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
            """Get optimizer."""
            optimizers = {
                'adam': optim.Adam(self.model.parameters(), lr=learning_rate),
                'sgd': optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9),
                'rmsprop': optim.RMSprop(self.model.parameters(), lr=learning_rate),
                'adamw': optim.AdamW(self.model.parameters(), lr=learning_rate),
            }
            return optimizers.get(optimizer_name, optim.Adam(self.model.parameters(), lr=learning_rate))

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Make predictions.

            Args:
                X: Input features

            Returns:
                Predictions
            """
            self.model.eval()

            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)

            return predicted.cpu().numpy()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """
            Get prediction probabilities.

            Args:
                X: Input features

            Returns:
                Prediction probabilities
            """
            self.model.eval()

            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                proba = torch.softmax(outputs, dim=1)

            return proba.cpu().numpy()

        def save_checkpoint(self, path: str):
            """
            Save model checkpoint.

            Args:
                path: Path to save checkpoint
            """
            checkpoint_path = Path(path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
            }, checkpoint_path)

            logger.info(f"Checkpoint saved to {checkpoint_path}")

        def load_checkpoint(self, path: str):
            """
            Load model checkpoint.

            Args:
                path: Path to checkpoint
            """
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)

            logger.info(f"Checkpoint loaded from {path}")


    def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                           batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create PyTorch data loaders.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            Train and validation data loaders
        """
        # Create training loader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # Create validation loader
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

else:
    # Dummy classes if PyTorch not available
    class FeedForwardNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

    class PyTorchTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Install with: pip install torch")

    def create_data_loaders(*args, **kwargs):
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
