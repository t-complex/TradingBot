import ta
import torch
import glob
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import tensorflow as tf
from joblib import dump
from scipy import signal
from fastdtw import fastdtw
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from shap import KernelExplainer
from darts.models import NHiTSModel
from hyperopt import hp, tpe, Trials
from Time2VecLayer import Time2VecLayer
from tensorflow.keras.callbacks import EarlyStopping
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import plot_acf, check_seasonality
from keras.models import Model
from keras.layers import Dense, Input , LSTM, Bidirectional
from pytorch_forecasting import TemporalFusionTransformer
from scipy.spatial.distance import euclidean
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from DataPreprocessing import DataPreprocessing
from tpot import TPOTRegressor  # Example using TPOT

class DataMainpulation:
    # def __init__(self)

    def split_data(self, X, y, n_splits, test_size):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X_train_full, X_test, y_train_full, y_test = [], [], [], []
        for train_index, test_index in tscv.split(X):
            X_train_full, X_test = X[train_index], X[test_index]
            y_train_full, y_test = y[train_index], y[test_index]
            # Create a validation set from the end of the training set
        val_size = int(test_size * len(X_train_full))
        X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
        y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scaling_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        scaler = StandardScaler()
        X_train_scaled, y_train_scaled = scaler.fit_transform(X_train), scaler.fit_transform(y_train.reshape(-1, 1))
        X_val_scaled, y_val_scaled = scaler.transform(X_val), scaler.transform(y_val.reshape(-1, 1))
        X_test_scaled, y_test_scaled = scaler.transform(X_test), scaler.transform(y_test.reshape(-1, 1))
        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled

    def define_difference_feature_importance(self, X_train, y_train):
        # Randomized search for RandomForestRegressor
        param_grid_RFR = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'bootstrap': [True, False]
        }
        random_search_RFR = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_grid_RFR, n_iter=20, cv=5)
        random_search_RFR.fit(X_train, y_train)
        best_params_RFR = random_search_RFR.best_params_
        feature_importances_RFR = random_search_RFR.best_estimator_.feature_importances_
        # Grid search for GradientBoostingRegressor
        param_grid_GBR = {
            'learning_rate': [0.01, 0.1, 0.5],
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search_GBR = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid_GBR, cv=5)
        grid_search_GBR.fit(X_train, y_train)
        best_params_GBR = grid_search_GBR.best_params_
        feature_importances_GBR = grid_search_GBR.best_estimator_.feature_importances_
        # Calculate permutation importances for RandomForestRegressor
        permutation_importances_RFR = permutation_importance(random_search_RFR, X_train, y_train, n_repeats=10)
        feature_importances_permutation_RFR = permutation_importances_RFR.importances_mean
        # Calculate permutation importances for GradientBoostingRegressor
        permutation_importances_GBR = permutation_importance(grid_search_GBR, X_train, y_train, n_repeats=10)
        feature_importances_permutation_GBR = permutation_importances_GBR.importances_mean
        # Calculate SHAP values for RandomForestRegressor
        explainer_RFR = KernelExplainer(random_search_RFR.predict, X_train)
        shap_values_RFR = explainer_RFR.shap_values(X_train)
        feature_importances_shap_RFR = np.abs(shap_values_RFR).mean(axis=0)
        # Calculate SHAP values for GradientBoostingRegressor
        explainer_GBR = KernelExplainer(grid_search_GBR.predict, X_train)
        shap_values_GBR = explainer_GBR.shap_values(X_train)
        feature_importances_shap_GBR = np.abs(shap_values_GBR).mean(axis=0)
        # Combine feature importances
        final_feature_importances = {
            'RandomForestRegressor': feature_importances_RFR,
            'GradientBoostingRegressor': feature_importances_GBR,
            'PermutationImportance_RFR': feature_importances_permutation_RFR,
            'PermutationImportance_GBR': feature_importances_permutation_GBR,
            'SHAP_RFR': feature_importances_shap_RFR,
            'SHAP_GBR': feature_importances_shap_GBR
        }
        return final_feature_importances

    def visualize_feature_importance(self, X_train, feature_importances_random, feature_importances_grid,
                                     feature_importances_permutation, feature_importances_shap):
        plt.figure(figsize=(10, 6))
        plt.bar(X_train.columns, feature_importances_random, label='Random Forest')
        plt.bar(X_train.columns, feature_importances_grid, label='Grid Search')
        plt.bar(X_train.columns, feature_importances_permutation, label='Permutation Importance')
        plt.bar(X_train.columns, feature_importances_shap, label='Kernel Explainer')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
