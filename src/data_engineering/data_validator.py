"""
Data validation and quality checks.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation and quality assessment."""

    def __init__(self, data: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize data validator.

        Args:
            data: DataFrame to validate
            config: Validation configuration
        """
        self.data = data
        self.config = config or {}
        self.validation_results = {}

    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            Dictionary of validation results
        """
        logger.info("Running comprehensive data validation...")

        self.validation_results = {
            'shape': self.check_shape(),
            'missing_values': self.check_missing_values(),
            'duplicates': self.check_duplicates(),
            'data_types': self.check_data_types(),
            'outliers': self.check_outliers(),
            'value_ranges': self.check_value_ranges(),
            'correlations': self.check_correlations(),
            'class_balance': self.check_class_balance(),
        }

        self._print_validation_summary()
        return self.validation_results

    def check_shape(self) -> Dict[str, int]:
        """Check data shape."""
        shape_info = {
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'is_empty': len(self.data) == 0
        }
        logger.info(f"Data shape: {self.data.shape}")
        return shape_info

    def check_missing_values(self) -> Dict[str, Any]:
        """Check for missing values."""
        missing_counts = self.data.isnull().sum()
        missing_pct = (missing_counts / len(self.data) * 100).round(2)

        missing_info = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict(),
            'columns_all_missing': missing_counts[missing_counts == len(self.data)].index.tolist()
        }

        if missing_info['total_missing'] > 0:
            logger.warning(f"Found {missing_info['total_missing']} missing values across {len(missing_info['columns_with_missing'])} columns")

        return missing_info

    def check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate rows."""
        n_duplicates = self.data.duplicated().sum()
        duplicate_info = {
            'n_duplicates': int(n_duplicates),
            'duplicate_percentage': round(n_duplicates / len(self.data) * 100, 2),
            'has_duplicates': n_duplicates > 0
        }

        if duplicate_info['has_duplicates']:
            logger.warning(f"Found {n_duplicates} duplicate rows ({duplicate_info['duplicate_percentage']}%)")

        return duplicate_info

    def check_data_types(self) -> Dict[str, Any]:
        """Check data types of columns."""
        dtype_info = {
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.data.select_dtypes(include=['datetime64']).columns.tolist(),
        }

        logger.info(f"Data types - Numeric: {len(dtype_info['numeric_columns'])}, "
                   f"Categorical: {len(dtype_info['categorical_columns'])}, "
                   f"Datetime: {len(dtype_info['datetime_columns'])}")

        return dtype_info

    def check_outliers(self, method: str = 'iqr') -> Dict[str, Any]:
        """
        Check for outliers in numeric columns.

        Args:
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')

        Returns:
            Outlier information
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_info = {}

        for col in numeric_cols:
            if method == 'iqr':
                outliers = self._detect_outliers_iqr(self.data[col])
            elif method == 'zscore':
                outliers = self._detect_outliers_zscore(self.data[col])
            else:
                continue

            if outliers.sum() > 0:
                outlier_info[col] = {
                    'n_outliers': int(outliers.sum()),
                    'outlier_percentage': round(outliers.sum() / len(self.data) * 100, 2)
                }

        if outlier_info:
            logger.warning(f"Found outliers in {len(outlier_info)} columns")

        return outlier_info

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series(z_scores > threshold, index=series.dropna().index)

    def check_value_ranges(self) -> Dict[str, Dict]:
        """Check value ranges for numeric columns."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        range_info = {}

        for col in numeric_cols:
            range_info[col] = {
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max()),
                'mean': float(self.data[col].mean()),
                'median': float(self.data[col].median()),
                'std': float(self.data[col].std()),
            }

        return range_info

    def check_correlations(self, threshold: float = 0.9) -> Dict[str, Any]:
        """
        Check for high correlations between features.

        Args:
            threshold: Correlation threshold

        Returns:
            Correlation information
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {'high_correlations': {}}

        corr_matrix = self.data[numeric_cols].corr().abs()

        # Get upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find high correlations
        high_corr = {}
        for column in upper.columns:
            high_corr_cols = upper[column][upper[column] > threshold]
            if len(high_corr_cols) > 0:
                high_corr[column] = high_corr_cols.to_dict()

        if high_corr:
            logger.warning(f"Found {len(high_corr)} features with high correlations (>{threshold})")

        return {'high_correlations': high_corr, 'threshold': threshold}

    def check_class_balance(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Check class balance for classification tasks.

        Args:
            target_col: Target column name

        Returns:
            Class balance information
        """
        if target_col is None or target_col not in self.data.columns:
            return {'message': 'No target column specified or found'}

        value_counts = self.data[target_col].value_counts()
        value_pct = (value_counts / len(self.data) * 100).round(2)

        balance_info = {
            'class_counts': value_counts.to_dict(),
            'class_percentages': value_pct.to_dict(),
            'is_balanced': value_pct.max() / value_pct.min() < 2 if len(value_pct) > 1 else True,
            'imbalance_ratio': float(value_pct.max() / value_pct.min()) if len(value_pct) > 1 else 1.0
        }

        if not balance_info['is_balanced']:
            logger.warning(f"Class imbalance detected. Ratio: {balance_info['imbalance_ratio']:.2f}")

        return balance_info

    def _print_validation_summary(self):
        """Print validation summary."""
        logger.info("=" * 50)
        logger.info("DATA VALIDATION SUMMARY")
        logger.info("=" * 50)

        # Shape
        logger.info(f"Dataset shape: {self.data.shape}")

        # Missing values
        missing = self.validation_results['missing_values']
        logger.info(f"Total missing values: {missing['total_missing']}")

        # Duplicates
        duplicates = self.validation_results['duplicates']
        logger.info(f"Duplicate rows: {duplicates['n_duplicates']}")

        # Outliers
        outliers = self.validation_results['outliers']
        if outliers:
            logger.info(f"Columns with outliers: {len(outliers)}")

        logger.info("=" * 50)


def validate_data(data: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to validate data.

    Args:
        data: DataFrame to validate
        config: Validation configuration

    Returns:
        Validation results
    """
    validator = DataValidator(data, config)
    return validator.validate_all()
