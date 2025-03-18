import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    def __init__(self):
        self.scalers = {}

    def preprocess(self, df, target_cols, sequence_length=60, train_test_split=0.8):
        """
        Preprocess time series data for model input.

        Args:
            df (pd.DataFrame): Input dataframe with time series data.
            target_cols (list): List of target columns to predict.
            sequence_length (int): Length of input sequences.
            train_test_split (float): Ratio of training data.

        Returns:
            dict: Preprocessed data including train/test splits and scalers.
        """
        logger.info(f"Preprocessing dataframe with shape {df.shape}")

        # Ensure DataFrame is sorted by date (if present)
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        # Create a copy to avoid modifying the original
        data = df.copy()

        # Handle missing values
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                logger.warning(f"Column {col} has {data[col].isnull().sum()} missing values. Filling with forward fill.")
                data[col] = data[col].fillna(method='ffill')

        # Scale the data
        for col in data.select_dtypes(include=[np.number]).columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[col] = scaler.fit_transform(data[[col]])
            self.scalers[col] = scaler

        # Create sequences
        X, y = self._create_sequences(data, target_cols, sequence_length)

        # Split data
        split_idx = int(len(X) * train_test_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Created train set with {len(X_train)} samples and test set with {len(X_test)} samples")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scalers': self.scalers,
            'feature_cols': data.columns.tolist(),
            'target_cols': target_cols
        }

    def _create_sequences(self, data, target_cols, sequence_length):
        """
        Create sequences for time series prediction.
        """
        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:i+sequence_length].values)

            # Use `.iloc` to correctly extract target values
            target_values = data.iloc[i+sequence_length][target_cols].values
            y.append(target_values if isinstance(target_values, np.ndarray) else [target_values])

        return np.array(X), np.array(y)

    def inverse_transform(self, predictions, target_cols):
        """
        Convert scaled predictions back to original scale.
        """
        inverse_preds = np.zeros_like(predictions)

        for i, col in enumerate(target_cols):
            if col in self.scalers:
                # Reshape for inverse transform
                col_preds = predictions[:, i].reshape(-1, 1)
                inverse_preds[:, i] = self.scalers[col].inverse_transform(col_preds).flatten()

        return inverse_preds