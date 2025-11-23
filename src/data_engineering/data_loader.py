"""
Data loading utilities for various file formats.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
import json
import pickle

logger = logging.getLogger(__name__)


class DataLoader:
    """Universal data loader for various file formats."""

    SUPPORTED_FORMATS = ['.csv', '.json', '.parquet', '.xlsx', '.feather', '.pkl', '.pickle']

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader.

        Args:
            data_path: Path to data file or directory
        """
        self.data_path = Path(data_path)

    def load(self, **kwargs) -> Union[pd.DataFrame, Dict, Any]:
        """
        Load data based on file extension.

        Args:
            **kwargs: Additional arguments to pass to specific loader

        Returns:
            Loaded data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        extension = self.data_path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")

        loader_map = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.xlsx': self._load_excel,
            '.feather': self._load_feather,
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle,
        }

        logger.info(f"Loading data from {self.data_path}")
        data = loader_map[extension](**kwargs)
        logger.info(f"Data loaded successfully. Shape: {self._get_shape(data)}")

        return data

    def _load_csv(self, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(self.data_path, **kwargs)

    def _load_json(self, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        # Try to convert to DataFrame if possible
        if isinstance(data, list):
            return pd.DataFrame(data)
        return data

    def _load_parquet(self, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(self.data_path, **kwargs)

    def _load_excel(self, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(self.data_path, **kwargs)

    def _load_feather(self, **kwargs) -> pd.DataFrame:
        """Load Feather file."""
        return pd.read_feather(self.data_path, **kwargs)

    def _load_pickle(self, **kwargs) -> Any:
        """Load pickled file."""
        with open(self.data_path, 'rb') as f:
            return pickle.load(f)

    def _get_shape(self, data: Any) -> str:
        """Get shape of data."""
        if isinstance(data, pd.DataFrame):
            return f"{data.shape}"
        elif isinstance(data, np.ndarray):
            return f"{data.shape}"
        elif isinstance(data, (list, dict)):
            return f"{len(data)} items"
        return "Unknown"

    def save(self, data: Any, output_path: Union[str, Path], **kwargs):
        """
        Save data to file.

        Args:
            data: Data to save
            output_path: Output file path
            **kwargs: Additional arguments for specific saver
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extension = output_path.suffix.lower()

        saver_map = {
            '.csv': lambda d, p: d.to_csv(p, index=False, **kwargs),
            '.json': lambda d, p: d.to_json(p, **kwargs) if isinstance(d, pd.DataFrame) else json.dump(d, open(p, 'w')),
            '.parquet': lambda d, p: d.to_parquet(p, **kwargs),
            '.xlsx': lambda d, p: d.to_excel(p, index=False, **kwargs),
            '.feather': lambda d, p: d.to_feather(p, **kwargs),
            '.pkl': lambda d, p: pickle.dump(d, open(p, 'wb')),
            '.pickle': lambda d, p: pickle.dump(d, open(p, 'wb')),
        }

        if extension not in saver_map:
            raise ValueError(f"Unsupported output format: {extension}")

        logger.info(f"Saving data to {output_path}")
        saver_map[extension](data, output_path)
        logger.info("Data saved successfully")


def load_data(path: Union[str, Path], **kwargs) -> Any:
    """
    Convenience function to load data.

    Args:
        path: Path to data file
        **kwargs: Additional arguments

    Returns:
        Loaded data
    """
    loader = DataLoader(path)
    return loader.load(**kwargs)


def save_data(data: Any, path: Union[str, Path], **kwargs):
    """
    Convenience function to save data.

    Args:
        data: Data to save
        path: Output path
        **kwargs: Additional arguments
    """
    loader = DataLoader(path)
    loader.save(data, path, **kwargs)
