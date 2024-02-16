"""
Risk Management and Strategy Optimization
Risk management is a critical component of successful trading.
We’ll explore various risk management techniques and how to optimize our strategies to achieve better performance.
# Implement a simple risk management technique by limiting the maximum position size
"""


from Backtest import Backtest
from MovingAverageCrossoverStrategy import MovingAverageCrossoverStrategy


class RiskManagedStrategy(MovingAverageCrossoverStrategy):
    def __init__(self, short_window, long_window, max_position_size):
        super.__init__(short_window, long_window)
        self.max_position_size = max_position_size

    def generate_signals(self, data):
        signals = super().generate_signals(data)
        signals['positions'] = signals['positions'].apply(lambda x: min(x, self.max_position_size))
        return signals

if __name__ == '__main__':
    # Optimize the strategy by adjusting the windows and position size
    optimized_strategy = RiskManagedStrategy(short_window=40, long_window=180, max_position_size=50)
    optimized_signals = optimized_strategy.generate_signals(data['JPM'])

    # Backtest the optimized strategy
    optimized_backtest = Backtest(data['JPM'], optimized_signals)
    optimized_portfolio = optimized_backtest.portfolio

    # Plot the optimized equity curve
    # plt.figure(figsize=(14, 7))
    # plt.plot(optimized_portfolio['total'], label='Optimized Portfolio Value')
    # plt.title('Optimized Portfolio Value Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value (USD)')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.close()

