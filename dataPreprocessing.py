"""
This file serve as the Data preprocessing, including handing missing values
Feature engineering and so on
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataPreprocessing:

    def handle_missing_values(self, data, strategy="knn"):
        if strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        elif strategy == "simple":
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        else:
            raise ValueError("Invalid strategy. Choose 'knn' or 'simple'.")
        return imputer.fit_transform(data)

    def remove_outliers(self, data):
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def normalize_data(self, data, scaler="minmax"):
        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers[scaler]
        return scaler.fit_transform(data)

if __name__ == '__main__':
    dp = DataPreprocessing()
    cryptos = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'ADA', 'DOGE']
    for crypto in cryptos:
        data = pd.read_csv(f'{crypto}-USD.csv')
        data = dp.handle_missing_values(data)
        data = dp.remove_outliers(data)
        data = dp.normalize_data(data)
        data.to_csv(f'{crypto}-Processed.csv')
