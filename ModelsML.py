"""
This file serve as the model building for the trading bot
This file will build 3 different model then ensemble them together
Models: [Prophet, NHiTSModel, Temporal Fusion Transformer, XGBoost]
"""

import glob
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from scipy import signal
from prophet import Prophet
from keras.models import Model
from xgboost import XGBRegressor
from darts.models import NHiTSModel
from hyperopt import hp, tpe, Trials
from Time2VecLayer import Time2VecLayer
# from mlxtend.classifier import StratifiedTimeSeriesSplit
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from darts.dataprocessing.transformers import Scaler
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from keras.layers import Dense, Input , LSTM, Bidirectional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelsML:
    def define_Prophet_model(self):
        # Prophet Model
        prophet_model = Prophet(seasonality_mode="multiplicative")
        prophet_model.add_seasonality(name="yearly", period=365, fourier_order=5)
        return prophet_model
    def define_NHiTS_model(self):
        nhits_model = NHiTSModel(input_chunk_length=168, output_chunk_length=120, random_state=42)
        return nhits_model
    def define_t2v_bilstm_timedistributed(self, train_data):
        inp = Input(shape=(train_data.shape[1], train_data.shape[2]))
        t2v_model = Time2VecLayer(100)(inp)
        t2v_model = Bidirectional(LSTM(100, activation='tanh', return_sequences = True))(t2v_model)
        t2v_model = Model(inp, t2v_model)
        t2v_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')
        return t2v_model
    def define_TFT_model(self, train_data):
        # Temporal Fusion Transformer
        transformer_model = TemporalFusionTransformer(
            input_shape=(train_data.shape[1], 1),
            make_timeseries=True,
            output_activation="linear",
            # Adjust other model parameters here
        )
        transformer_model.compile(loss="mse", optimizer="adam")
        return transformer_model
    def define_XGBoost_model(self):
        # XGBoost
        xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        xgb_model.compile(loss="mse")
        return xgb_model
    def fitting_models(self, X_train, X_val, y_train, y_val, test_data, scaled_train,
                       scaled_val, prophet_model, nhits_model, t2v_model , tft_model, xgb_model):
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        for _ in range(3):
            prophet_data = pd.DataFrame({'ds': X_train['date'], 'y': y_train})
            prophet_model.fit(prophet_data, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
            nhits_model.fit(scaled_train, validation_data=scaled_val, epochs=50, callbacks=[early_stopping])
            t2v_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
            tft_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], epochs=50, early_stopping_rounds=10)
        # Predict each model separately
        train_scaler = StandardScaler()
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        prophet_predictions = prophet_model.predict(future)["yhat"].squeeze()
        scaled_pred_nhits = nhits_model.predict(n=120)
        nhits_predictions = train_scaler.inverse_transform(scaled_pred_nhits)
        scaled_pred_t2v = t2v_model.predict(n=120)
        t2v_predictions = train_scaler.inverse_transform(scaled_pred_t2v)
        tft_predictions = tft_model.predict(test_data)["prediction"]
        xgb_predictions = xgb_model.predict(test_data)
        return prophet_predictions, nhits_predictions, t2v_predictions, tft_predictions, xgb_predictions
    def define_ensemble_model(self, prophet_predictions, nhits_predictions, t2v_predictions,
                              tft_predictions, xgb_predictions):
        ensemble_model = StackingRegressor(
            estimators=[
                ('prophet', prophet_predictions),
                ('nhits_lstm', nhits_predictions),
                ('transformer', tft_predictions),
                ('Time2Vec', t2v_predictions),
                ('xgb', xgb_predictions)
            ],
            final_estimator=Dense(1),
            cv=5
        )
        return ensemble_model
    def tune_hyperparameters(self, ensemble_model, X_train, y_train):
        hyperparameters = {
            'stackingregressor__passthrough': hp.choice('passthrough', [True, False]),
            'stackingregressor__final_estimator__n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'prophet_model': {
                'seasonality_periods': [1, 3, 6, 12],
                'holiday_seasonality': True,
                ' Autoregressive_integrated_moving_average': True,
                'change_point_prior_probability': 0.5,
                'n_change_points': 5,
                'mcmc_samples': 100,
                'warmup_steps': 50,
                'steps': 50,
                'seed': 42
            },
            'nhits_model': {
                'units': [20, 30, 40, 50],
                'return_sequences': [True, False],
                'num_heads': [4, 6, 8, 10],
                'hidden_size': [20, 30, 40, 50],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.3, 0.5],
                'attention_dropout': [0.1, 0.3, 0.5],
                'kernel_size': [3, 5, 7],
                'stride': [2, 3, 4],
                'padding': [1, 2, 3],
                'activation': ['relu', 'tanh', 'sigmoid']
            },
            't2v_model': {
                'embedding_dim': hp.choice('embedding_dim', [16, 32, 64]),
                'activation_original': hp.choice('activation_original', ['relu', 'tanh', 'leaky_relu']),
                'activation_transformed': hp.choice('activation_transformed', ['relu', 'tanh', 'leaky_relu']),
                'weight_initializer': hp.choice('weight_initializer',
                                                ['glorot_uniform', 'he_uniform', 'xavier_uniform']),
            },
            'transformer_model': {
                'num_heads': [4, 6, 8, 10],
                'hidden_size': [20, 30, 40, 50],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.3, 0.5],
                'attention_dropout': [0.1, 0.3, 0.5],
                'kernel_size': [3, 5, 7],
                'stride': [2, 3, 4],
                'padding': [1, 2, 3],
                'activation': ['relu', 'tanh', 'sigmoid']
            },
            'xgb_model': {
                'num_leaves': [31, 62, 127],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.5],
                'n_estimators': [500, 1000, 2000],
                'n_jobs': [-1, 2, 4]
            }
        }
        trials = Trials()
        best_params = tpe.Trials(trials=trials).suggest(
            lambda x: -mean_squared_error(y_train, ensemble_model.fit(X_train, y_train).predict(X_train)),
            search_space=hyperparameters, max_evals=100)
        # Train the ensemble with best hyperparameters
        ensemble_model.set_params(**best_params)
        final_ensemble_model_predictions_with_best_param = ensemble_model.fit(X_train, y_train).predict(test_data)
        return final_ensemble_model_predictions_with_best_param, mean_squared_error
    def evaluate_model(self, model, y_test, y_pred, crypto):
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='True Values')
        mape = np.mean(np.abs(y_test - y_pred) / y_test) * 100
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {mean_squared_error(y_test, y_pred)}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred)}") # the lower the MAE, the better the model predicts
        print(f"MAPE: {mape}")
        print(f"R-squared: {r2}")
        dump(model, f'{model}.joblib')
        submission_df = pd.DataFrame({"id": test_data, "target": model.flatten()})
        submission_df.to_csv(f'{crypto}-{model}-submission.csv', index=False)
        # Visualize prediction errors
        plt.hist(y_test - y_pred)
        plt.xlabel("Prediction Error")
        plt.ylabel("Number of Samples")
        plt.title("Prediction Error Distribution")
        plt.show()