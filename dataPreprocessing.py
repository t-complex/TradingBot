"""
This file serve as the Data preprocessing, including handing missing values
Feature engineering and so on
"""
import glob
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats

class DataPreprocessing:

    def __init__(self, handle_missing_values='knn', remove_outliers_threshold=3, scaler='minmax'):
        self.handle_missing_values_strategy = handle_missing_values
        self.remove_outliers_threshold = remove_outliers_threshold
        self.scaler = scaler

    def handle_missing_values(self, data):
        numeric_data = data.select_dtypes(include=[np.number])
        if self.handle_missing_values_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif self.handle_missing_values_strategy == 'simple':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else:
            raise ValueError("Invalid strategy. Choose 'knn' or 'simple'.")
        imputed_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
        data[numeric_data.columns] = imputed_data
        return data

    def remove_outliers(self, data):
        numeric_data = data.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_data))
        mask = (z_scores < self.remove_outliers_threshold).all(axis=1)
        return data[mask]

    def normalize_data(self, data):
        numeric_data = data.select_dtypes(include=[np.number])
        scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}
        selected_scaler = scalers[self.scaler]
        scaled_data = pd.DataFrame(selected_scaler.fit_transform(numeric_data), columns=numeric_data.columns)
        data[numeric_data.columns] = scaled_data
        return data

    def preprocess_data(self, data):
        data = self.handle_missing_values(data)
        data = self.remove_outliers(data)
        data = self.normalize_data(data)
        return data


if __name__ == '__main__':
    dp = DataPreprocessing(handle_missing_values='knn',
                           remove_outliers_threshold=3,
                           scaler='minmax')
    cryptos = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'ADA', 'DOGE']
    files = glob.glob('data/*.csv')
    # data = pd.read_csv(files[0], parse_dates=['date'])
    # print(data.columns)
    for file in files:
        crypto = file.split('\\')[1].split('-')[0]
        data = pd.read_csv(file, parse_dates=['date'])
        df = dp.preprocess_data(data)
        df.to_csv(f'data/pre/{crypto}.csv', index=False)  # Avoid saving index column
        print(f'{crypto} data preprocessed and saved successfully.')

