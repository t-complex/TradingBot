"""
This file serve as the Data preprocessing, including handing missing values
Feature engineering and so on
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from autoimpute.imputations import MiceImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataPreprocessing:
    @staticmethod
    def handle_missing_values(data, strategy="knn"):
        if strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        elif strategy == "mice":
            imputer = MICEImputer(n_imputations=5)
        else:
            raise ValueError("Invalid strategy. Choose 'knn' or 'mice'.")
        imputer.fit(data)
        data = imputer.transform(data)
        return data
    @staticmethod
    def remove_outliers(data, iqr_multiplier=1.5):
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]
    @staticmethod
    def create_features(data):
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()
        data["RSI"] = data.calculate_rsi(data["Close"])
    @staticmethod
    def normalize_data(data, scaler="minmax"):
        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers[scaler]
        scaler.fit(data)
        return scaler.transform(data)
    @staticmethod
    def split_data(data):
        time_series_split = TimeSeriesSplit(n_splits=3)
        train_data, val_data, test_data = [], [], []
        for train_index, val_index, test_index in time_series_split.split(data):
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]
            test_data = data.iloc[test_index]
        return train_data, val_data, test_data
    @staticmethod
    def preprocess_data(data_path):
        data = pd.read_csv(data_path)
        DataPreprocessing.handle_missing_values(data)
        data = DataPreprocessing.remove_outliers(data)
        DataPreprocessing.create_features(data)
        normalized_data = DataPreprocessing.normalize_data(data, scaler="minmax")
        train_data, val_data, test_data = DataPreprocessing.split_data(normalized_data)
        return train_data, val_data, test_data

    @staticmethod
    def calculate_rsi(close_prices, period=14):
        """
        Calculates the Relative Strength Index (RSI) for a given set of closing prices.
        Args:
            close_prices: A list of closing prices.
            period: The RSI calculation period (default: 14).
        Returns:
            A list of RSI values for each closing price.
        """
        if len(close_prices) < period:
            raise ValueError("Not enough data for RSI calculation.")
        rsi = []
        for i in range(period):
            rsi.append(np.nan)  # Handle initial RSI values
        up_changes = [0] * (period - 1)
        down_changes = [0] * (period - 1)
        for i in range(1, len(close_prices)):
            change = close_prices[i] - close_prices[i - 1]
            if change > 0:
                up_changes.append(change)
                down_changes.append(0)
            else:
                up_changes.append(0)
                down_changes.append(abs(change))
        avg_gain = np.mean(up_changes[period:])
        avg_loss = np.mean(down_changes[period:])
        if avg_loss == 0:
            rsi.extend([100] * (len(close_prices) - period))
        else:
            rsi.extend([100 - (100 / (1 + avg_gain / avg_loss)) for _ in range(len(close_prices) - period)])
        return rsi