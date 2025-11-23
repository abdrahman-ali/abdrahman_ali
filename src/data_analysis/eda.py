"""
Exploratory Data Analysis (EDA) utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EDA:
    """Comprehensive Exploratory Data Analysis toolkit."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDA toolkit.

        Args:
            data: DataFrame to analyze
        """
        self.data = data

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report.

        Returns:
            Dictionary containing all EDA results
        """
        logger.info("Generating EDA report...")

        report = {
            'basic_info': self.basic_info(),
            'statistical_summary': self.statistical_summary(),
            'missing_values': self.missing_values_analysis(),
            'cardinality': self.cardinality_analysis(),
            'correlations': self.correlation_analysis(),
            'distributions': self.distribution_analysis(),
        }

        self._print_report(report)
        return report

    def basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'column_names': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
        }

    def statistical_summary(self) -> Dict[str, pd.DataFrame]:
        """Get statistical summary of the dataset."""
        numeric_summary = self.data.describe()
        categorical_summary = self.data.describe(include=['object', 'category'])

        return {
            'numeric': numeric_summary,
            'categorical': categorical_summary
        }

    def missing_values_analysis(self) -> Dict[str, Any]:
        """Analyze missing values."""
        missing_counts = self.data.isnull().sum()
        missing_pct = (missing_counts / len(self.data) * 100).round(2)

        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_pct
        }).sort_values('missing_count', ascending=False)

        return {
            'total_missing': int(missing_counts.sum()),
            'missing_by_column': missing_df[missing_df['missing_count'] > 0].to_dict(),
            'columns_with_all_missing': missing_counts[missing_counts == len(self.data)].index.tolist()
        }

    def cardinality_analysis(self) -> Dict[str, int]:
        """Analyze cardinality of categorical columns."""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        cardinality = {}

        for col in categorical_cols:
            cardinality[col] = self.data[col].nunique()

        return dict(sorted(cardinality.items(), key=lambda x: x[1], reverse=True))

    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlations between numeric features.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return pd.DataFrame()

        return self.data[numeric_cols].corr(method=method)

    def distribution_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze distributions of numeric features."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        distributions = {}

        for col in numeric_cols:
            distributions[col] = {
                'mean': float(self.data[col].mean()),
                'median': float(self.data[col].median()),
                'std': float(self.data[col].std()),
                'skewness': float(self.data[col].skew()),
                'kurtosis': float(self.data[col].kurtosis()),
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max()),
                'range': float(self.data[col].max() - self.data[col].min()),
            }

        return distributions

    def value_counts_analysis(self, column: str, top_n: int = 10) -> pd.Series:
        """
        Get value counts for a specific column.

        Args:
            column: Column name
            top_n: Number of top values to return

        Returns:
            Value counts
        """
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found in data")

        return self.data[column].value_counts().head(top_n)

    def outlier_summary(self, method: str = 'iqr') -> Dict[str, int]:
        """
        Get summary of outliers in numeric columns.

        Args:
            method: Outlier detection method

        Returns:
            Outlier counts by column
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.data[col] < Q1 - 1.5 * IQR) | (self.data[col] > Q3 + 1.5 * IQR)).sum()
            else:  # zscore
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = (z_scores > 3).sum()

            if outliers > 0:
                outlier_counts[col] = int(outliers)

        return outlier_counts

    def _print_report(self, report: Dict[str, Any]):
        """Print EDA report summary."""
        logger.info("=" * 60)
        logger.info("EXPLORATORY DATA ANALYSIS REPORT")
        logger.info("=" * 60)

        basic_info = report['basic_info']
        logger.info(f"Dataset shape: ({basic_info['n_rows']}, {basic_info['n_columns']})")
        logger.info(f"Memory usage: {basic_info['memory_usage_mb']:.2f} MB")

        missing = report['missing_values']
        logger.info(f"Total missing values: {missing['total_missing']}")

        logger.info("=" * 60)


def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to analyze data.

    Args:
        data: DataFrame to analyze

    Returns:
        EDA report
    """
    eda = EDA(data)
    return eda.generate_report()
