"""
Building Trading Strategies
Now, let’s start building our trading strategies.
We’ll create a simple moving average crossover strategy and backtest it to evaluate its performance.
"""
import math

import pandas as pd
from termcolor import colored as cl
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
class BacktestStrategy:
    def __init__(self, data):
        self.data = data
    def backtest(self, data):
        strategy_returns = data['close'].pct_change()
        strategy_cumulative_returns = (1 + strategy_returns).cumprod()
        strategy_cumulative_returns.plot(figsize=(10, 6))
        plt.title('Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.show()
    def analyze(self, data):
        # Implement statistical analysis logic here
        # Example: Calculate descriptive statistics
        statistics = data['Indicator'].describe()
        print(statistics)
    def sensitivity_analysis(self, data):
        # Implement sensitivity analysis logic here
        # Example: Vary parameter values and observe results
        pass
    def test_and_optimize(self):
        # Implement testing and optimization logic here
        self.test_specific_parameters(parameters={'parameter1': value1, 'parameter2': value2})
        self.optimize_parameters()
    def test_specific_parameters(self, parameters):
        # Implement testing specific parameter logic here
        # Example: Test specific parameter values
        # (Replace this with your specific indicator testing logic)
        for param in ParameterGrid(parameters):
            print(f"Testing parameter: {param}")
            # Perform testing with parameter values
            # Calculate performance metrics and store results
    def evaluate_strategy(self):
        # Implement strategy evaluation logic here
        # Example: Evaluate strategy performance on test data
        pass
    def optimize_parameters(self):
        # Implement parameter optimization logic here
        # Example: Use grid search or optimization algorithms to find best parameters
        pass
    def implement_strategy(self, data, investment):
        in_position, equity, no_of_shares = False, investment, 0
        for i in range(1, len(data)):
            if data['SMA_8'][i - 1] < data['SMA_13'][i - 1] and data['SMA_8'][i] > data['SMA_13'][i] and data['SMA_5'][i] > data['SMA_8'][i] and data['close'][i] > data['SMA_5'][i] and in_position == False:
                no_of_shares = math.floor(equity / data.close[i])
                equity -= (no_of_shares * data.close[i])
                in_position = True
                print(cl('BUY: ', color='green', attrs=['bold']),
                      f'{no_of_shares} Shares are bought at ${data.close[i]} on {str(data["date"][i])[:10]}')
            elif data['SMA_8'][i - 1] > data['SMA_13'][i - 1] and data['SMA_8'][i] < data['SMA_13'][i] and data['SMA_5'][i] < data['SMA_8'][i] and data['close'][i] < data['SMA_5'][i] and in_position == True:
                equity += (no_of_shares * data.close[i])
                in_position = False
                print(cl('SELL: ', color='red', attrs=['bold']),
                      f'{no_of_shares} Shares are bought at ${data.close[i]} on {str(data["date"][i])[:10]}')
        if in_position == True:
            equity += (no_of_shares * data.close[i])
            print(cl(f'\nClosing position at {data.close[i]} on {str(data["date"][i])[:10]}', attrs=['bold']))
            in_position = False

        earning = round(equity - investment, 2)
        roi = round(earning / investment * 100, 2)
        print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs=['bold']))
    def backtest_advance_indicator(self):
        df = self.data.dropna().reset_index(drop=True)
        shifted_features = []
        for i in range(20):
            for feature in ['Close', 'Volume', 'BOLU', 'RSI', '%K', '%D', 'EMA_10', 'EMA_20', 'ATR', 'ADX']:
                shifted_feature = df[feature].shift(i)
                shifted_feature.name = f'{feature}_{i}d_ago'
                shifted_features.append(shifted_feature)
        df_shifted = pd.concat(shifted_features, axis=1)
        df = pd.concat([df, df_shifted], axis=1)
        return df
