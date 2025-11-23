"""
Data preprocessing and feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DataPipeline:
    """Comprehensive data preprocessing and feature engineering pipeline."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPipeline':
        """
        Fit the pipeline on training data.

        Args:
            X: Training features
            y: Training target

        Returns:
            Fitted pipeline
        """
        logger.info("Fitting data pipeline...")

        self.feature_names = X.columns.tolist()

        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit imputers
        if numeric_features:
            self.imputers['numeric'] = SimpleImputer(strategy='mean')
            self.imputers['numeric'].fit(X[numeric_features])

        if categorical_features:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'].fit(X[categorical_features])

        # Fit scalers for numeric features
        if numeric_features:
            scaler_type = self.config.get('scaling', {}).get('method', 'standard')
            self.scalers['numeric'] = self._get_scaler(scaler_type)
            X_numeric_imputed = self.imputers['numeric'].transform(X[numeric_features])
            self.scalers['numeric'].fit(X_numeric_imputed)

        # Fit encoders for categorical features
        if categorical_features:
            encoder_type = self.config.get('encoding', {}).get('method', 'onehot')
            self.encoders['categorical'] = self._get_encoder(encoder_type)

            X_categorical_imputed = self.imputers['categorical'].transform(X[categorical_features])

            if encoder_type == 'onehot':
                self.encoders['categorical'].fit(X_categorical_imputed)
            else:
                for i, col in enumerate(categorical_features):
                    encoder = self._get_encoder(encoder_type)
                    encoder.fit(X_categorical_imputed[:, i])
                    self.encoders[col] = encoder

        self.is_fitted = True
        logger.info("Pipeline fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        logger.info("Transforming data...")

        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        transformed_parts = []

        # Transform numeric features
        if numeric_features and 'numeric' in self.imputers:
            X_numeric = X[numeric_features].copy()
            X_numeric_imputed = self.imputers['numeric'].transform(X_numeric)
            X_numeric_scaled = self.scalers['numeric'].transform(X_numeric_imputed)

            numeric_df = pd.DataFrame(
                X_numeric_scaled,
                columns=numeric_features,
                index=X.index
            )
            transformed_parts.append(numeric_df)

        # Transform categorical features
        if categorical_features and 'categorical' in self.imputers:
            X_categorical = X[categorical_features].copy()
            X_categorical_imputed = self.imputers['categorical'].transform(X_categorical)

            encoder_type = self.config.get('encoding', {}).get('method', 'onehot')

            if encoder_type == 'onehot' and 'categorical' in self.encoders:
                X_categorical_encoded = self.encoders['categorical'].transform(X_categorical_imputed)

                if hasattr(X_categorical_encoded, 'toarray'):
                    X_categorical_encoded = X_categorical_encoded.toarray()

                # Get feature names
                if hasattr(self.encoders['categorical'], 'get_feature_names_out'):
                    cat_feature_names = self.encoders['categorical'].get_feature_names_out(categorical_features)
                else:
                    cat_feature_names = [f"{col}_{i}" for col in categorical_features
                                        for i in range(X_categorical_encoded.shape[1] // len(categorical_features))]

                categorical_df = pd.DataFrame(
                    X_categorical_encoded,
                    columns=cat_feature_names,
                    index=X.index
                )
                transformed_parts.append(categorical_df)
            else:
                # Label encoding or other encoders
                encoded_data = []
                for i, col in enumerate(categorical_features):
                    if col in self.encoders:
                        encoded_col = self.encoders[col].transform(X_categorical_imputed[:, i])
                        encoded_data.append(encoded_col)

                categorical_df = pd.DataFrame(
                    np.column_stack(encoded_data),
                    columns=categorical_features,
                    index=X.index
                )
                transformed_parts.append(categorical_df)

        # Combine all transformed features
        if transformed_parts:
            X_transformed = pd.concat(transformed_parts, axis=1)
        else:
            X_transformed = X.copy()

        logger.info(f"Data transformed. Output shape: {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform data.

        Args:
            X: Data to fit and transform
            y: Target variable

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def _get_scaler(self, scaler_type: str):
        """Get scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer(),
        }
        return scalers.get(scaler_type, StandardScaler())

    def _get_encoder(self, encoder_type: str):
        """Get encoder based on type."""
        encoders = {
            'onehot': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        }
        return encoders.get(encoder_type, OneHotEncoder(sparse_output=False, handle_unknown='ignore'))


class FeatureEngineer:
    """Feature engineering utilities."""

    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between feature pairs.

        Args:
            df: Input dataframe
            feature_pairs: List of feature pairs to create interactions

        Returns:
            DataFrame with interaction features
        """
        df_copy = df.copy()

        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_copy[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                logger.info(f"Created interaction feature: {feat1}_x_{feat2}")

        return df_copy

    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            df: Input dataframe
            features: Features to create polynomials
            degree: Polynomial degree

        Returns:
            DataFrame with polynomial features
        """
        df_copy = df.copy()

        for feat in features:
            if feat in df.columns:
                for d in range(2, degree + 1):
                    df_copy[f'{feat}_pow_{d}'] = df[feat] ** d
                    logger.info(f"Created polynomial feature: {feat}_pow_{d}")

        return df_copy

    @staticmethod
    def create_binned_features(df: pd.DataFrame, features: List[str], n_bins: int = 5) -> pd.DataFrame:
        """
        Create binned features.

        Args:
            df: Input dataframe
            features: Features to bin
            n_bins: Number of bins

        Returns:
            DataFrame with binned features
        """
        df_copy = df.copy()

        for feat in features:
            if feat in df.columns:
                df_copy[f'{feat}_binned'] = pd.cut(df[feat], bins=n_bins, labels=False)
                logger.info(f"Created binned feature: {feat}_binned")

        return df_copy


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target
        test_size: Test set size
        val_size: Validation set size
        random_state: Random state
        stratify: Whether to stratify split

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")

    stratify_y = y if stratify else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    # Second split: train vs val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_y_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=stratify_y_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = X_train.iloc[:0], y_train.iloc[:0]  # Empty validation set

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
