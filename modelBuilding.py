"""
This file serve as the model building for the trading bot
This file will build 3 different model then ensemble them together
Models: [Prophet, Bi-LSTM with MultiHeadAttention, Temporal Fusion Transformer, XGBoost]
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, MultiHeadAttention
from pytorch_forecasting import TemporalFusionTransformer
from xgboost import XGBRegressor
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
from dataPreprocessing import DataPreprocessing
import matplotlib.pyplot as plt

class ModelBuilding:
    def load_data(self, file_path):
        train_data, val_data, test_data = DataPreprocessing.preprocess_data(file_path)
        return train_data, val_data, test_data
    def define_models(self, train_data, val_data, test_data, crypto):
        # Example using feature importances from a tree-based model
        features, target = ['open', 'high', 'low', 'Volume USD', f'Volume {crypto}'], 'close'
        forest = RandomForestRegressor()
        forest.fit(train_data[features], train_data[target])
        importances = forest.feature_importances_
        important_features = features[np.argsort(importances)[-5:]]  # Select top 5 features
        price_features = [feature for feature in important_features if feature in ['open', 'high', 'low']]
        volume_features = [feature for feature in important_features if feature not in price_features]
        target_variable = target

        # Prophet Model
        prophet_model = Prophet(seasonality_mode="multiplicative")
        prophet_model.add_seasonality(name="yearly", period=365, fourier_order=5)
        # Bi-LSTM with MultiHeadAttention
        bi_lstm_model = Sequential()
        bi_lstm_model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(train_data.shape[1], 1)))
        bi_lstm_model.add(MultiHeadAttention(num_heads=8))
        bi_lstm_model.add(Bidirectional(LSTM(32)))
        bi_lstm_model.add(Dense(1))
        bi_lstm_model.compile(loss="mse", optimizer="adam")
        # Temporal Fusion Transformer
        transformer_model = TemporalFusionTransformer(
            input_shape=(train_data.shape[1], 1),
            make_timeseries=True,
            output_activation="linear",
            # Adjust other model parameters here
        )
        transformer_model.compile(loss="mse", optimizer="adam")
        # XGBoost
        xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        xgb_model.compile(loss="mse")
        # Fit individual models with early stopping while Repeat cross-validation for robustness
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        X_train_for_ensemble_model, y_train_for_ensemble_model = [], []
        for _ in range(3):
            X_train, X_val, y_train, y_val = train_test_split(train_data, test_size=0.2, random_state=42)
            X_train_for_ensemble_model, y_train_for_ensemble_model = X_train, y_train
            X_train, y_train = train_data[price_features, volume_features], train_data[target_variable]
            prophet_data = pd.DataFrame({'ds': X_train['date'], 'y': y_train})
            prophet_model.fit(prophet_data)
            bi_lstm_model.fit(
                X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[early_stopping]
            )
            transformer_model.fit(
                X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[early_stopping]
            )
            xgb_model.fit(
                X_train[price_features, volume_features],
                y_train,
                early_stopping_rounds=10,
                eval_set=[(X_val[price_features, volume_features], y_val)]
            )

        # Predict each model separately
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        prophet_predictions = prophet_model.predict(future)["yhat"].squeeze()
        lstm_predictions = bi_lstm_model.predict(test_data)
        transformer_predictions = transformer_model.predict(test_data)["prediction"]
        xgb_predictions = xgb_model.predict(test_data[price_features, volume_features])
        # Evaluate each model
        prophet_rmse = mean_squared_error(target_variable , prophet_predictions, squared=False)
        bi_lstm_model_rmse = mean_squared_error(target_variable, lstm_predictions, squared=False)
        transformer_rmse = mean_squared_error(target_variable, transformer_predictions, squared=False)
        xgb_rmse = mean_squared_error(target_variable, xgb_predictions, squared=False)
        # Print RMSE for each model
        print(f"Prophet RMSE: {prophet_rmse}")
        print(f"Bi-LSTM RMSE: {bi_lstm_model_rmse}")
        print(f"Temporal Fusion RMSE: {transformer_rmse}")
        print(f"XGBoost RMSE: {xgb_rmse}")
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(target_variable, label='True Values')
        plt.plot(prophet_predictions, label='Prophet Forecast')
        plt.plot(lstm_predictions, label='Bi-LSTM Forecast')
        plt.plot(transformer_predictions, label='Temporal Fusion Forecast')
        plt.plot(xgb_predictions, label='XGBoost Forecast')
        plt.legend()
        plt.show()

        # Define ensemble model using StackingRegressor
        ensemble_model = StackingRegressor(
            estimators=[
                ('prophet', prophet_model),
                ('bi-lstm', bi_lstm_model),
                ('transformer', transformer_model),
                ('xgb', xgb_model)
            ],
            final_estimator=Dense(1),
            cv=5
        )

        # Combine predictions (adjust weights as needed)
        ensemble_predictions = np.average(
            [prophet_predictions, lstm_predictions, transformer_predictions, xgb_predictions],
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
        grid_search.fit(train_data[price_features, volume_features], train_data[target_variable])
        predictions = grid_search.predict(test_data[price_features, volume_features])
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        ensemble_model.set_params(**grid_search.best_params_)
        ensemble_model.fit(X_train_for_ensemble_model, y_train_for_ensemble_model)

        # Save individual models and the ensemble model
        dump(prophet_predictions, "prophet_model.joblib")
        dump(lstm_predictions, "lstm_model.joblib")
        dump(transformer_predictions, "transformer_model.joblib")
        dump(xgb_predictions, "xgb_model.joblib")
        dump(ensemble_model, "ensemble_model.joblib")
        # Load models later
        loaded_prophet = load("prophet_model.joblib")
        loaded_ensemble = load("ensemble_model.joblib")
        # Plot the performance of the models
        plt.figure(figsize=(12, 6))
        plt.plot(test_data['Close'], label="True")
        plt.plot(ensemble_predictions, label="Ensemble")
        plt.legend()
        plt.title("Performance of Models")
        plt.show()
        # Save the results
        submission_df = pd.DataFrame({"id": test_data["id"], "target": ensemble_predictions.flatten()})
        submission_df.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    mb = ModelBuilding()
    train_BTC_data, val_BTC_data, test_BTC_data = mb.load_data("data/BTC-USD.csv")
    train_ETH_data, val_ETH_data, test_ETH_data = mb.load_data("data/ETH-USD.csv")

    mb.define_models(train_BTC_data, val_BTC_data, test_BTC_data, "BTC")
    mb.define_models(train_ETH_data, val_ETH_data, test_ETH_data, "ETH")
