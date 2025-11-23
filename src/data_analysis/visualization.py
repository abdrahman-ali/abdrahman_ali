"""
Data visualization utilities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DataVisualizer:
    """Comprehensive data visualization toolkit."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "docs/reports/figures"):
        """
        Initialize data visualizer.

        Args:
            data: DataFrame to visualize
            output_dir: Directory to save plots
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_missing_values(self, save: bool = True) -> plt.Figure:
        """
        Plot missing values heatmap.

        Args:
            save: Whether to save the plot

        Returns:
            Figure object
        """
        logger.info("Creating missing values plot...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate missing values
        missing = self.data.isnull()

        if missing.sum().sum() == 0:
            logger.info("No missing values to plot")
            plt.close()
            return None

        # Plot heatmap
        sns.heatmap(missing, yticklabels=False, cbar=True, cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Samples', fontsize=12)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'missing_values.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_distributions(self, columns: Optional[List[str]] = None, save: bool = True) -> plt.Figure:
        """
        Plot distributions of numeric columns.

        Args:
            columns: Columns to plot (if None, plot all numeric)
            save: Whether to save the plot

        Returns:
            Figure object
        """
        logger.info("Creating distribution plots...")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(3, len(columns))
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, col in enumerate(columns):
            if col in self.data.columns:
                ax = axes[idx]
                self.data[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)

        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'distributions.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_correlation_matrix(self, save: bool = True, method: str = 'pearson') -> plt.Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            save: Whether to save the plot
            method: Correlation method

        Returns:
            Figure object
        """
        logger.info("Creating correlation matrix...")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation matrix")
            return None

        corr_matrix = self.data[numeric_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'correlation_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_boxplots(self, columns: Optional[List[str]] = None, save: bool = True) -> plt.Figure:
        """
        Plot boxplots for numeric columns.

        Args:
            columns: Columns to plot
            save: Whether to save the plot

        Returns:
            Figure object
        """
        logger.info("Creating boxplots...")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(3, len(columns))
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, col in enumerate(columns):
            if col in self.data.columns:
                ax = axes[idx]
                self.data.boxplot(column=col, ax=ax)
                ax.set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
                ax.set_ylabel(col, fontsize=10)

        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'boxplots.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_pairplot(self, columns: Optional[List[str]] = None,
                     hue: Optional[str] = None, save: bool = True) -> sns.PairGrid:
        """
        Create pairplot for features.

        Args:
            columns: Columns to include
            hue: Column to use for color coding
            save: Whether to save the plot

        Returns:
            PairGrid object
        """
        logger.info("Creating pairplot...")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to 5 features

        data_subset = self.data[columns + ([hue] if hue and hue not in columns else [])]

        pairplot = sns.pairplot(data_subset, hue=hue, diag_kind='hist', plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16, fontweight='bold')

        if save:
            save_path = self.output_dir / 'pairplot.png'
            pairplot.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return pairplot

    def plot_target_distribution(self, target_col: str, save: bool = True) -> plt.Figure:
        """
        Plot target variable distribution.

        Args:
            target_col: Target column name
            save: Whether to save the plot

        Returns:
            Figure object
        """
        logger.info(f"Creating target distribution plot for {target_col}...")

        if target_col not in self.data.columns:
            logger.error(f"Column {target_col} not found")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if self.data[target_col].dtype in [np.number]:
            # Numeric target - histogram
            self.data[target_col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency', fontsize=12)
        else:
            # Categorical target - bar plot
            value_counts = self.data[target_col].value_counts()
            value_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Count', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        ax.set_title(f'Distribution of {target_col}', fontsize=16, fontweight='bold')
        ax.set_xlabel(target_col, fontsize=12)

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'target_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray,
                               top_n: int = 20, save: bool = True) -> plt.Figure:
        """
        Plot feature importances.

        Args:
            feature_names: List of feature names
            importances: Array of importance values
            top_n: Number of top features to plot
            save: Whether to save the plot

        Returns:
            Figure object
        """
        logger.info("Creating feature importance plot...")

        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.barh(importance_df['feature'], importance_df['importance'], edgecolor='black', alpha=0.7)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()

        if save:
            save_path = self.output_dir / 'feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig

    def generate_all_plots(self, target_col: Optional[str] = None):
        """
        Generate all standard plots.

        Args:
            target_col: Target column name (optional)
        """
        logger.info("Generating all standard plots...")

        self.plot_missing_values()
        self.plot_distributions()
        self.plot_correlation_matrix()
        self.plot_boxplots()

        if target_col and target_col in self.data.columns:
            self.plot_target_distribution(target_col)

        logger.info(f"All plots saved to {self.output_dir}")


def visualize_data(data: pd.DataFrame, output_dir: str = "docs/reports/figures",
                  target_col: Optional[str] = None):
    """
    Convenience function to generate all visualizations.

    Args:
        data: DataFrame to visualize
        output_dir: Output directory
        target_col: Target column name
    """
    visualizer = DataVisualizer(data, output_dir)
    visualizer.generate_all_plots(target_col)
