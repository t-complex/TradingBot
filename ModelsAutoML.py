"""
This file is an alternative approach to builing the 3-models using AutoML
"""

import h2o
import time
import glob
import numpy as np
import pandas as pd
import DataPreprocessing
# import alternativeApproach
from tpot import TPOTRegressor
from h2o.automl import H2OAutoML
from sklearn.model_selection import TimeSeriesSplit
from autogluon.tabular import TabularPredictor
# from auto-sklearn.regression import AutoSklearnRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

class ModelsAutoML:
    def __init__(self, X, y, n_splits=5, test_size=0.2):
        self.X, self.y = X, y
        self.n_splits, self.test_size = n_splits, test_size
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
    def split_data(self):
        for train_index, test_index in self.tscv.split(self.X):
            self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]
    def define_tpot_model(self):
        self.tpot_model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
        self.tpot_model.fit(self.X_train, self.y_train)
        self.tpot_best_pipeline = self.tpot_model.fitted_pipeline_
        self.tpot_predictions = self.tpot_model.predict(self.X_test)
        self.evaluate_model(self.tpot_predictions, self.tpot_best_pipeline, "TPOT")
    def define_h20_model(self):
        h2o.init()
        self.h2o_model = H2OAutoML(max_models=20, seed=1)
        self.h2o_model.train(self.X_train.columns.tolist(), self.y_train.name, training_frame=self.X_train.join(self.y_train))
        self.h2o_best_pipeline = self.h2o_model.get_best_model()
        self.h2o_predictions = self.h2o_model.predict(self.X_test)
        self.evaluate_model(self.h2o_predictions.as_data_frame().values.flatten(), self.h2o_best_pipeline, "H2O")
    def define_gluon_model(self):
        self.gluon_model = TabularPredictor(label=self.y_train.name).fit(train_data=self.X_train.join(self.y_train))
        self.gluon_predictions = self.gluon_model.predict(self.X_test)
        self.gluon_best_pipeline = self.gluon_model.leaderboard[0]['model']  # Access the best model
        self.evaluate_model(self.gluon_predictions.values, self.gluon_best_pipeline, "AutoGluon")
    # def define_sklearn_model(self):
    #     self.sklearn_model = AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
    #     self.sklearn_model.fit(self.X_train, self.y_train)
    #     self.sklearn_best_pipeline = self.sklearn_model.fitted_pipeline_
    #     self.sklearn_predictions = self.sklearn_model.predict(self.X_test)
    #     self.evaluate_model(self.sklearn_predictions, self.sklearn_best_pipeline, "Auto-sklearn")
    def evaluate_model(self, predictions, best_pipeline, model_name):
        metrics = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, predictions)),
            "MAE": mean_absolute_error(self.y_test, predictions),
            "MAPE": mean_absolute_percentage_error(self.y_test - predictions),
            "R-squared": r2_score(self.y_test, predictions),
        }
        print(f"{model_name} - Best Pipeline: {best_pipeline}")
        for metric_name, value in metrics.items():
            print(f"\t{metric_name}: {value}")
        # Example usage in compare_models:
        models = {
            "TPOT": self.tpot_best_pipeline,
            "H2O": self.h2o_best_pipeline,
            "AutoGluon": self.gluon_best_pipeline,
        }
        for model_name, pipeline in models.items():
            self.evaluate_model(self.tpot_predictions, pipeline, model_name)
    def compare_models(self):
        self.split_data()
        self.define_tpot_model()
        self.define_h20_model()
        self.define_gluon_model()
        # self.define_sklearn_model()
        # Find the best model based on the evaluation metrics
        # best_model = max([self.tpot_model, self.h2o_model, self.gluon_model, self.sklearn_model],
        #                  key=lambda model: model.score(self.X_test, self.y_test))
        best_model = max([self.tpot_model, self.h2o_model, self.gluon_model],
                         key=lambda model: model.score(self.X_test, self.y_test))
        best_model.fit(self.X_train, self.y_train)
        final_predictions = best_model.predict(self.X_test)
        best_model.save('best_model.pkl')
        self.evaluate_model(final_predictions, best_model, "Best Model")

if __name__ == '__main__':
    files = glob.glob('data/*.csv')
    file = files[0]
    crypto = file.split('.')[0].split('\\')[1]
    data = pd.read_csv(file)
    pd.to_datetime(data['date'])
    # print(data.columns.values)
    dp = DataPreprocessing.DataPreprocessing(data)
    dp.handle_missing_values()
    dp.remove_outliers()
    dp.normalize_data()
    preprocessed_data = dp.get_preprocessed_data()
    y, X = pd.DataFrame(preprocessed_data['close']), preprocessed_data.drop('close', axis=1)
    model = ModelsAutoML(X, y)
    model.compare_models()


