



from sklearn.model_selection import train_test_split
from dataPreprocessing import DataPreprocessing
from tpot import TPOTRegressor  # Example using TPOT
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class ModelBuildingAutoML:

    def load_data(self, file_path):
        train_data, val_data, test_data = DataPreprocessing.preprocess_data(file_path)
        return train_data, val_data, test_data

    def define_AutoML(self, train_data, val_data, test_data, crypto):
        # Data splitting and preparation
        X_train, X_val, y_train, y_val = train_test_split(train_data, test_size=0.2, random_state=42)

        # Feature selection (optional, adjust as needed)
        features = ['open', 'high', 'low', 'Volume USD', f'Volume {crypto}']
        target_variable = 'close'

        # AutoML pipeline
        automl = TPOTRegressor(generations=5, population_size=20, verbosity=2)  # Adjust parameters as needed
        automl.fit(X_train[features], y_train)

        # Best pipeline and predictions
        best_pipeline = automl.fitted_pipeline_
        predictions = best_pipeline.predict(test_data[features])

        # Evaluation metrics
        rmse = mean_squared_error(test_data[target_variable], predictions, squared=False)
        mae = mean_absolute_error(test_data[target_variable], predictions)
        mape = mean_absolute_percentage_error(test_data[target_variable], predictions)
        r2 = r2_score(test_data[target_variable], predictions)
        print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R-squared: {r2}")

        # Save best pipeline
        automl.export('tpot_pipeline.py')  # Example for TPOT

    def complete_pipeline(self, file_path, crypto):
        # Combine data loading, preprocessing, modeling, and evaluation
        data = self.load_data(file_path)
        train_data, val_data, test_data = DataPreprocessing.preprocess_data(data)
        self.define_AutoML(train_data, val_data, test_data, crypto)

if __name__ == '__main__':
    mb = ModelBuildingAutoML()
    train_BTC_data, val_BTC_data, test_BTC_data = mb.load_data("data/BTC-USD.csv")
    train_ETH_data, val_ETH_data, test_ETH_data = mb.load_data("data/ETH-USD.csv")

    mb.define_AutoML(train_BTC_data, val_BTC_data, test_BTC_data, "BTC")
    mb.define_AutoML(train_ETH_data, val_ETH_data, test_ETH_data, "ETH")