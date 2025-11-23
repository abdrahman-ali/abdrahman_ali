"""Tests for data loader module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data_engineering.data_loader import DataLoader, load_data, save_data


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_csv(self, sample_dataframe, temp_dir):
        """Test loading CSV files."""
        # Save test CSV
        csv_path = temp_dir / 'test.csv'
        sample_dataframe.to_csv(csv_path, index=False)

        # Load
        loader = DataLoader(csv_path)
        loaded_data = loader.load()

        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_dataframe.shape

    def test_save_csv(self, sample_dataframe, temp_dir):
        """Test saving CSV files."""
        csv_path = temp_dir / 'test_save.csv'

        loader = DataLoader(csv_path)
        loader.save(sample_dataframe, csv_path)

        assert csv_path.exists()

        # Load and verify
        loaded_data = pd.read_csv(csv_path)
        assert loaded_data.shape == sample_dataframe.shape

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DataLoader('nonexistent.csv')

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_convenience_functions(self, sample_dataframe, temp_dir):
        """Test convenience functions."""
        csv_path = temp_dir / 'test_convenience.csv'

        # Save using convenience function
        save_data(sample_dataframe, csv_path)
        assert csv_path.exists()

        # Load using convenience function
        loaded_data = load_data(csv_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == sample_dataframe.shape
