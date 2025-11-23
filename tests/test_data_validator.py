"""Tests for data validator module."""
import pytest
import pandas as pd
import numpy as np

from src.data_engineering.data_validator import DataValidator, validate_data


class TestDataValidator:
    """Test cases for DataValidator class."""

    @pytest.fixture
    def clean_dataframe(self):
        """Create clean dataframe for testing."""
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })

    @pytest.fixture
    def dirty_dataframe(self):
        """Create dataframe with issues for testing."""
        return pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'd']  # Has duplicate
        })

    def test_validate_clean_data(self, clean_dataframe):
        """Test validation on clean data."""
        validator = DataValidator(clean_dataframe)
        results = validator.validate_all()

        assert results['missing_values']['total_missing'] == 0
        assert results['duplicates']['n_duplicates'] == 0

    def test_validate_dirty_data(self, dirty_dataframe):
        """Test validation on data with issues."""
        validator = DataValidator(dirty_dataframe)
        results = validator.validate_all()

        assert results['missing_values']['total_missing'] > 0
        assert 'A' in results['missing_values']['columns_with_missing']

    def test_check_shape(self, clean_dataframe):
        """Test shape checking."""
        validator = DataValidator(clean_dataframe)
        shape_info = validator.check_shape()

        assert shape_info['n_rows'] == 5
        assert shape_info['n_columns'] == 3
        assert shape_info['is_empty'] is False

    def test_check_data_types(self, clean_dataframe):
        """Test data type checking."""
        validator = DataValidator(clean_dataframe)
        dtype_info = validator.check_data_types()

        assert 'numeric_columns' in dtype_info
        assert 'categorical_columns' in dtype_info
        assert len(dtype_info['numeric_columns']) == 2
        assert len(dtype_info['categorical_columns']) == 1

    def test_convenience_function(self, clean_dataframe):
        """Test convenience function."""
        results = validate_data(clean_dataframe)

        assert 'missing_values' in results
        assert 'duplicates' in results
        assert 'data_types' in results
