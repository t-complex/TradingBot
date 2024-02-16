"""
Backtesting and Strategy Evaluation
Backtesting is the process of testing a trading strategy on historical data to assess its viability.
We’ll write a backtesting engine in Python and evaluate our moving average crossover strategy.

"""
import numpy as np
import pandas as pd


class Backtest:

    def __init__(self, data, signals, initial_capital=100000.0):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()
        self.portfolio = self.backtest_portfolio()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['JPM'] = 100 * self.signals['signal']   # This is a simple example with a fixed number of shares
        return positions

    def backtest_portfolio(self):
        portfolio = self.positions.multiply(self.data['Close'], axis=0)
        pos_diff = self.positions.diff()
        portfolio['holdings'] = (self.positions.multiply(self.data['Close'], axis=0)).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff.multiply(self.data['Close'], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

if __name__ == '__main__':
    # Backtest the strategy
    backtest = Backtest(data['JPM'], signals)
    portfolio = backtest.portfolio
    returns = portfolio['returns']
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
    print(sharpe_ratio)

    # Plot the equity curve
    # plt.figure(figsize=(14, 7))
    # plt.plot(portfolio['total'], label='Portfolio Value')
    # plt.title('Portfolio Value Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value (USD)')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.close()