"""
This file serve as the Data preprocessing, including handing missing values
Feature engineering and so on
"""

import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing:
    def __init__(self, data):
        self.data = data
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.numerical_data = self.data.drop('date', axis=1)
    def handle_missing_values(self):
        imputer = KNNImputer(n_neighbors=2)
        self.numerical_data = pd.DataFrame(imputer.fit_transform(self.numerical_data),
                                           columns=self.numerical_data.columns)
    def remove_outliers(self):
        z_scores, threshold = stats.zscore(self.numerical_data), 3
        self.numerical_data = self.numerical_data[(z_scores < threshold).all(axis=1)]
    def normalize_data(self):
        scaler = MinMaxScaler()
        self.numerical_data = pd.DataFrame(scaler.fit_transform(self.numerical_data),
                                           columns=self.numerical_data.columns)
    def get_preprocessed_data(self):
        return pd.concat([self.data['date'], self.numerical_data], axis=1)

