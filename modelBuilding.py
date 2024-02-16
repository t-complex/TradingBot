"""
This file serve as the model building for the trading bot
This file will build 3 different model then ensemble them together
Models: [Prophet, NHiTSModel, Temporal Fusion Transformer, XGBoost]
"""

import numpy as np
import pandas as pd
from joblib import dump
from prophet import Prophet
from keras.layers import Dense
from pytorch_forecasting import TemporalFusionTransformer
from xgboost import XGBRegressor
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class ModelBuilding:

    def define_feature_importance(self, data, crypto):
        # Example using feature importances from a tree-based model
        features, target = ['open', 'high', 'low', 'Volume USD', f'Volume {crypto}'], 'close'
        forest = RandomForestRegressor()
        forest.fit(data[features], data[target])
        importances = forest.feature_importances_
        important_features = [features[i] for i in np.argsort(importances)[-5:]]  # Select top 5 features
        X, y = data[important_features], data[target]
        return X, y

    def split_data(self, X, y, test_size=0.2):
        time_series_split = TimeSeriesSplit(n_splits=3)
        X_train_full, X_test, y_train_full, y_test = [], [], [], []
        for train_index, test_index in time_series_split.split(X):
            X_train_full, X_test = X[train_index], X[test_index]
            y_train_full, y_test = y[train_index], y[test_index]
        # Create a validation set from the end of the training set
        val_size = int(test_size * len(X_train_full))
        X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
        y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
        test_data = [X_test, y_test]
        return X_train, X_val, y_train, y_val, test_data

    def define_Prophet_model(self):
        # Prophet Model
        prophet_model = Prophet(seasonality_mode="multiplicative")
        prophet_model.add_seasonality(name="yearly", period=365, fourier_order=5)
        return prophet_model

    def define_NHiTS_model(self):
        nhits_model = NHiTSModel(input_chunk_length=168, output_chunk_length=120, random_state=42)
        return nhits_model

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

    def fitting_models(self, X_train, X_val, y_train, y_val, test_data, scaled_train, scaled_val, prophet_model, nhits_model, tft_model, xgb_model):
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        for _ in range(3):
            prophet_data = pd.DataFrame({'ds': X_train['date'], 'y': y_train})
            prophet_model.fit(prophet_data, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
            nhits_model.fit(scaled_train, validation_data=scaled_val, epochs=50, callbacks=[early_stopping])
            tft_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], epochs=50, early_stopping_rounds=10)
        # Predict each model separately
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        prophet_predictions = prophet_model.predict(future)["yhat"].squeeze()
        scaled_pred_nhits = nhits_model.predict(n=120)
        nhits_predictions = train_scaler.inverse_transform(scaled_pred_nhits)
        tft_predictions = tft_model.predict(test_data)["prediction"]
        xgb_predictions = xgb_model.predict(test_data)
        return prophet_predictions, nhits_predictions, tft_predictions, xgb_predictions

    def define_ensemble_model(self, X_train, y_train, test_data, prophet_predictions,
                              nhits_predictions, tft_predictions, xgb_predictions, crypto):
        # Define ensemble model using StackingRegressor
        ensemble_model = StackingRegressor(
            estimators=[
                ('prophet', prophet_model),
                ('nhits_lstm', nhits_model),
                ('transformer', tft_model),
                ('xgb', xgb_model)
            ],
            final_estimator=Dense(1),
            cv=5
        )

        # Combine predictions (adjust weights as needed)
        ensemble_predictions = np.average(
            [prophet_predictions, nhits_predictions, tft_predictions, xgb_predictions],
            axis=0,
            weights=[0.3, 0.25, 0.2, 0.25])
        # Evaluate ensemble model performance, Calculate and print additional metrics
        rmse = mean_squared_error(test_data['Close'], ensemble_predictions)
        mae = mean_absolute_error(test_data['Close'], ensemble_predictions)
        mape = mean_absolute_percentage_error(test_data['Close'], ensemble_predictions)
        r2 = r2_score(test_data['Close'], ensemble_predictions)
        print(f"Ensemble RMSE: {rmse}", f"MAE: {mae}, MAPE: {mape}, R-squared: {r2}")

        # Define Hyperparameter Grids (Note: Use the correct model names)
        hyperparameters = {
            'stackingregressor__passthrough': {True, False},
            'stackingregressor__final_estimator__n_estimators': {100, 200, 500},
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
            'model': {
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
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(
            estimator=ensemble_model,
            param_grid=hyperparameters,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        # Fit and evaluate ensemble model
        grid_search.fit(X_train, y_train)
        predictions = grid_search.predict(test_data)
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        ensemble_model.set_params(**grid_search.best_params_)
        ensemble_model.fit(X_train, y_train)

        # Save individual models and the ensemble model
        dump(prophet_predictions, f'{crypto}-prophet_model.joblib')
        dump(nhits_predictions, f'{crypto}-nhits_model.joblib')
        dump(tft_predictions, f'{crypto}-tft_model.joblib')
        dump(xgb_predictions, f'{crypto}-xgb_model.joblib')
        dump(ensemble_model, f'{crypto}-ensemble_model.joblib')

        return predictions


if __name__ == '__main__':
    mb = ModelBuilding()
    cryptos = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'ADA', 'DOGE']

    for crypto in cryptos:
        data = pd.read_csv(f'{crypto}-USD.csv')
        X, y = mb.define_feature_importance(data, crypto)
        X_train, X_val, y_train, y_val, test_data = mb.split_data(X, y)
        train_data, target = [X_train, y_train], data['close']
        prophet_model = mb.define_Prophet_model()
        train_scaler, validate_scaler = Scaler(), Scaler()
        scaled_train = train_scaler.fit_transform(train_data)
        scaled_val = validate_scaler.fit_transform([X_val, y_val])
        nhits_model = mb.define_NHiTS_model()
        tft_model = mb.define_TFT_model(train_data)
        xgb_model = mb.define_XGBoost_model()
        (prophet_predictions, nhits_predictions,
         tft_predictions, xgb_predictions) = mb.fitting_models(X_train, X_val, y_train, y_val, test_data,
                                                               scaled_train, scaled_val, prophet_model,
                                                               nhits_model, tft_model, xgb_model)

        # Evaluate each model
        prophet_rmse = mean_squared_error(target, prophet_predictions, squared=False)
        bi_lstm_model_rmse = mean_squared_error(target, nhits_predictions, squared=False)
        transformer_rmse = mean_squared_error(target, tft_predictions, squared=False)
        xgb_rmse = mean_squared_error(target, xgb_predictions, squared=False)
        # Print RMSE for each model
        print(f"Prophet RMSE: {prophet_rmse}")
        print(f"Bi-LSTM RMSE: {bi_lstm_model_rmse}")
        print(f"Temporal Fusion RMSE: {transformer_rmse}")
        print(f"XGBoost RMSE: {xgb_rmse}")
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(target, label='True Values')
        plt.plot(prophet_predictions, label='Prophet Forecast')
        plt.plot(nhits_predictions, label='Time2Vec-BiLSTM Forecast')
        plt.plot(tft_predictions, label='Temporal Fusion Forecast')
        plt.plot(xgb_predictions, label='XGBoost Forecast')
        plt.legend()
        plt.show()

        ensemble_model_predictions = mb.define_ensemble_model(X_train, y_train, test_data, prophet_predictions,
                                 nhits_predictions, tft_predictions, xgb_predictions, crypto)

        # Load models later
        # loaded_prophet = load("prophet_model.joblib")
        # loaded_ensemble = load("ensemble_model.joblib")
        # Plot the performance of the models
        plt.figure(figsize=(12, 6))
        plt.plot(test_data, label="True")
        plt.plot(ensemble_model_predictions, label="Ensemble")
        plt.legend()
        plt.title("Performance of Models")
        plt.show()
        # Save the results
        submission_df = pd.DataFrame({"id": test_data, "target": ensemble_model_predictions.flatten()})
        submission_df.to_csv(f'{crypto}-submission.csv', index=False)




