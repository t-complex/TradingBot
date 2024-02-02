"""
This file serve as the model building for the trading bot
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Attention, concatenate
from pytorch_forecasting import TemporalFusionTransformer
from xgboost import XGBClassifier, XGBRegressor
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
        important_features = features[np.argsort(importances)[-5:]]  # Select top 5 features (adjust number if needed)

        price_features = [feature for feature in important_features if feature in ['open', 'high', 'low']]
        volume_features = [feature for feature in important_features if feature not in price_features]
        target_variable = target

        # Define individual models
        prophet_model = Prophet(seasonality_mode="multiplicative")
        prophet_model.add_seasonality(name="yearly", period=365, fourier_order=5)
        bi_lstm_model = Sequential()
        bi_lstm_model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(train_data.shape[1], 1)))
        bi_lstm_model.add(Attention())
        bi_lstm_model.add(Bidirectional(LSTM(32)))
        bi_lstm_model.add(Dense(1))
        bi_lstm_model.compile(loss="mse", optimizer="adam")
        transformer_model = TemporalFusionTransformer(
            input_shape=(train_data.shape[1], 1),
            make_timeseries=True,
            output_activation="linear",
            # Adjust other model parameters here
        )
        transformer_model.compile(loss="mse", optimizer="adam")
        xgb_model = XGBClassifier(objective="reg:squarederror", n_estimators=1000)
        xgb_model.compile(loss="mse")
        # Define an ensemble model using StackingRegressor
        ensemble_model = StackingRegressor(
            estimators=[
                ('prophet', prophet_model),
                ('lstm', bi_lstm_model),
                ('transformer', transformer_model),
                ('xgb', xgb_model)
            ],
            final_estimator=Dense(1),
            cv=5
        )

        # Early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)

        for _ in range(3):
            # Repeat cross-validation for robustness
            X_train, X_val, y_train, y_val = train_test_split(train_data, test_size=0.2, random_state=42)

            # Fit individual models with early stopping
            X_train, y_train = train_data[price_features, volume_features], train_data[target_variable]
            prophet_data = pd.DataFrame({'ds': X_train['date'], 'y': y_train})
            prophet_model.fit(prophet_data)
            bi_lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[early_stopping])
            transformer_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10,
                                  callbacks=[early_stopping])
            xgb_model.fit(X_train[price_features, volume_features], y_train, early_stopping_rounds=10, eval_set=[(X_val[price_features, volume_features], y_val)])

        # Ensemble Predictions and Interpretability
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        prophet_predictions = prophet_model.predict(future)["yhat"].squeeze()
        lstm_predictions = bi_lstm_model.predict(test_data)
        transformer_predictions = transformer_model.predict(test_data)["prediction"]
        xgb_predictions = xgb_model.predict(test_data[price_features, volume_features])

        # Combine predictions (adjust weights as needed)
        ensemble_predictions = np.average(
            [prophet_predictions, lstm_predictions, transformer_predictions, xgb_predictions],
            axis=0,
            weights=[0.3, 0.25, 0.2, 0.25]
        )

        # Interpretability: Prophet component analysis, SHAP values for XGBoost, etc.
        print(f"Ensemble RMSE: {mean_squared_error(test_data['Close'], ensemble_predictions)}")

        # Evaluate and compare individual and ensemble model performance with relevant metrics
        # Define Individual Model Estimators
        prophet_estimator = Prophet(seasonality_mode="multiplicative")
        prophet_estimator.add_seasonality(name="yearly", period=365, fourier_order=5)

        bi_lstm_estimator = Sequential()
        bi_lstm_estimator.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(train_data.shape[1], 1)))
        bi_lstm_estimator.add(Attention())
        bi_lstm_estimator.add(Bidirectional(LSTM(32)))
        bi_lstm_estimator.add(Dense(1))

        transformer_estimator = TemporalFusionTransformer(
            input_shape=(train_data.shape[1], 1),
            make_timeseries=True,
            output_activation="linear",
            # Adjust other model parameters here
        )

        xgb_estimator = XGBRegressor(objective="reg:squarederror", n_estimators=1000)

        # Define Hyperparameter Grids (Note: Use the correct model names)
        hyperparameter_grid = {
            "stackingregressor__passthrough": [True, False],
            "stackingregressor__final_estimator__n_estimators": [100, 200, 500],
            "prophet_estimator__changepoint_prior_scale": [0.001, 0.01, 0.1],  # Example for Prophet hyperparameter
            "lstm_estimator__dropout": [0.1, 0.2, 0.3],  # Example for LSTM hyperparameter
            # Add more hyperparameters relevant to other individual models and the meta-learner
        }

        # Ensemble Model and Grid Search
        ensemble_model = Sequential()
        ensemble_model.add(concatenate([prophet_estimator, bi_lstm_estimator, transformer_estimator, xgb_estimator]))
        ensemble_model.add(Dense(1))

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(
            estimator=ensemble_model,
            param_grid=hyperparameter_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        # Train the ensemble model
        grid_search.fit(train_data, train_data['Close'])

        # Make predictions
        predictions = grid_search.predict(test_data)
        # Print best parameters and score
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        ensemble_model.set_params(**grid_search.best_params_)
        ensemble_model.fit(X_train, y_train)

        # Calculate and print additional metrics
        mae = mean_absolute_error(test_data['Close'], ensemble_predictions)
        mape = mean_absolute_percentage_error(test_data['Close'], ensemble_predictions)
        r2 = r2_score(test_data['Close'], ensemble_predictions)
        print(f"MAE: {mae}, MAPE: {mape}, R-squared: {r2}")

        # Evaluate and compare individual and ensemble model performance
        prophet_predictions = prophet_model.predict(future)["yhat"].squeeze()
        lstm_predictions = bi_lstm_model.predict(test_data)
        transformer_predictions = transformer_model.predict(test_data)["prediction"]
        xgb_predictions = xgb_model.predict(test_data)

        # Combine predictions (adjust weights as needed)
        ensemble_predictions = np.average(
            [prophet_predictions, lstm_predictions, transformer_predictions, xgb_predictions],
            axis=0,
            weights=[0.3, 0.25, 0.2, 0.25]
        )

        # Evaluate ensemble model performance
        ensemble_rmse = mean_squared_error(test_data['Close'], ensemble_predictions)
        print(f"Ensemble RMSE: {ensemble_rmse}")

        # Save individual models
        dump(prophet_estimator, "prophet_model.joblib")
        dump(bi_lstm_estimator, "lstm_model.joblib")
        dump(transformer_estimator, "transformer_model.joblib")
        dump(xgb_estimator, "xgb_model.joblib")
        # Save the ensemble model
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


