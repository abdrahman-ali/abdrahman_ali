"""
TensorFlow/Keras deep learning trainer.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


if TENSORFLOW_AVAILABLE:
    def create_feedforward_model(input_size: int, hidden_layers: list, output_size: int,
                                 dropout: float = 0.2, batch_norm: bool = True,
                                 activation: str = 'relu') -> keras.Model:
        """
        Create feedforward neural network.

        Args:
            input_size: Input dimension
            hidden_layers: List of hidden layer sizes
            output_size: Output dimension
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            activation: Activation function

        Returns:
            Keras model
        """
        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_size,)))

        # Hidden layers
        for hidden_size in hidden_layers:
            model.add(layers.Dense(hidden_size))
            if batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation))
            model.add(layers.Dropout(dropout))

        # Output layer
        if output_size == 1:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(output_size, activation='softmax'))

        return model


    class TensorFlowTrainer:
        """TensorFlow/Keras model trainer."""

        def __init__(self, model: keras.Model, config: Optional[Dict] = None):
            """
            Initialize TensorFlow trainer.

            Args:
                model: Keras model
                config: Training configuration
            """
            self.model = model
            self.config = config or {}
            self.history = None

        def compile_model(self, optimizer: str = 'adam', learning_rate: float = 0.001,
                         loss: str = 'sparse_categorical_crossentropy',
                         metrics: list = None):
            """
            Compile the model.

            Args:
                optimizer: Optimizer name
                learning_rate: Learning rate
                loss: Loss function
                metrics: List of metrics
            """
            if metrics is None:
                metrics = ['accuracy']

            # Get optimizer
            optimizer_dict = {
                'adam': keras.optimizers.Adam(learning_rate=learning_rate),
                'sgd': keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
                'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate),
                'adamw': keras.optimizers.AdamW(learning_rate=learning_rate),
            }

            opt = optimizer_dict.get(optimizer, keras.optimizers.Adam(learning_rate=learning_rate))

            self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
            logger.info("Model compiled successfully")

        def train(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 epochs: int = 100, batch_size: int = 32,
                 callbacks_list: Optional[list] = None) -> keras.callbacks.History:
            """
            Train the model.

            Args:
                X_train: Training features
                y_train: Training labels
                X_val: Validation features
                y_val: Validation labels
                epochs: Number of epochs
                batch_size: Batch size
                callbacks_list: List of callbacks

            Returns:
                Training history
            """
            logger.info(f"Starting training for {epochs} epochs...")

            if callbacks_list is None:
                callbacks_list = self._get_default_callbacks()

            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)

            # Train
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )

            logger.info("Training completed!")
            return self.history

        def _get_default_callbacks(self) -> list:
            """Get default callbacks."""
            callback_list = []

            # Early stopping
            if self.config.get('early_stopping', {}).get('enabled', True):
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping', {}).get('patience', 10),
                    restore_best_weights=True,
                    verbose=1
                )
                callback_list.append(early_stop)

            # Reduce learning rate
            if self.config.get('reduce_lr', {}).get('enabled', True):
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('reduce_lr', {}).get('factor', 0.5),
                    patience=self.config.get('reduce_lr', {}).get('patience', 5),
                    min_lr=1e-7,
                    verbose=1
                )
                callback_list.append(reduce_lr)

            # Model checkpoint
            if self.config.get('model_checkpoint', {}).get('enabled', True):
                checkpoint_path = Path('models/checkpoints/best_model.h5')
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint = callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                callback_list.append(checkpoint)

            return callback_list

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Make predictions.

            Args:
                X: Input features

            Returns:
                Predictions
            """
            predictions = self.model.predict(X)

            if predictions.shape[1] == 1:
                return (predictions > 0.5).astype(int).flatten()
            else:
                return np.argmax(predictions, axis=1)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """
            Get prediction probabilities.

            Args:
                X: Input features

            Returns:
                Prediction probabilities
            """
            return self.model.predict(X)

        def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
            """
            Evaluate the model.

            Args:
                X_test: Test features
                y_test: Test labels

            Returns:
                Evaluation metrics
            """
            results = self.model.evaluate(X_test, y_test, verbose=0)

            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = results[i]

            logger.info("Evaluation results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            return metrics

        def save_model(self, path: str):
            """
            Save the model.

            Args:
                path: Path to save the model
            """
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")

        def load_model(self, path: str):
            """
            Load a model.

            Args:
                path: Path to the model
            """
            self.model = keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")


    def create_cnn_model(input_shape: tuple, num_classes: int,
                        num_filters: list = None, kernel_size: int = 3,
                        pool_size: int = 2, dense_units: list = None,
                        dropout: float = 0.2) -> keras.Model:
        """
        Create CNN model for image data.

        Args:
            input_shape: Input shape
            num_classes: Number of classes
            num_filters: List of filter sizes for conv layers
            kernel_size: Kernel size
            pool_size: Pool size
            dense_units: Dense layer units
            dropout: Dropout rate

        Returns:
            Keras CNN model
        """
        if num_filters is None:
            num_filters = [32, 64, 128]
        if dense_units is None:
            dense_units = [128, 64]

        model = models.Sequential()

        # Conv layers
        for i, filters in enumerate(num_filters):
            if i == 0:
                model.add(layers.Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape))
            else:
                model.add(layers.Conv2D(filters, kernel_size, activation='relu'))
            model.add(layers.MaxPooling2D(pool_size))
            model.add(layers.Dropout(dropout))

        # Flatten
        model.add(layers.Flatten())

        # Dense layers
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))

        return model


    def create_lstm_model(input_shape: tuple, num_classes: int,
                         lstm_units: list = None, dense_units: list = None,
                         dropout: float = 0.2) -> keras.Model:
        """
        Create LSTM model for sequence data.

        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of classes
            lstm_units: List of LSTM layer units
            dense_units: Dense layer units
            dropout: Dropout rate

        Returns:
            Keras LSTM model
        """
        if lstm_units is None:
            lstm_units = [64, 32]
        if dense_units is None:
            dense_units = [32]

        model = models.Sequential()

        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(layers.LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(dropout))

        # Dense layers
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))

        return model

else:
    # Dummy functions if TensorFlow not available
    def create_feedforward_model(*args, **kwargs):
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

    def create_cnn_model(*args, **kwargs):
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

    def create_lstm_model(*args, **kwargs):
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

    class TensorFlowTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
