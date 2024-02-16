"""
Building Trading Strategies
Now, let’s start building our trading strategies.
We’ll create a simple moving average crossover strategy and backtest it to evaluate its performance.

"""


import pandas as pd
import numpy as np


class MovingAverageCrossoverStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1, center=False).mean()
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:]
                                                         > signals['long_mavg'][self.short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        signals['RSI'] = self.calculate_rsi(data['Close'])
        return signals

    # # Function to fetch candlestick data:
    # def fetch_data(symbol, interval, lookback):
    #     bars = client.futures_historical_klines(symbol, interval, lookback)
    #     df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close'])
    #     df['close'] = pd.to_numeric(df['close'])
    #     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #     return df[['timestamp', 'close']]  # We return only what we'll need from the data
    #
    # # Main strategy logic & execution 👇
    # def sma_strategy(symbol='BTCUSDT', interval='1h', short_window=50, long_window=200, lookback='30 days ago UTC'):
    #     data = fetch_data(symbol, interval, lookback)
    #
    #     data['short_sma'] = data['close'].rolling(window=short_window).mean()
    #     data['long_sma'] = data['close'].rolling(window=long_window).mean()
    #
    #     # Assuming you're starting without an open position
    #     in_position = False
    #
    #     # Check for SMA crossover
    #     # If SMA crosses LMA Going short on the crypto)👇
    #     if data['short_sma'].iloc[-2] < data['long_sma'].iloc[-2] and data['short_sma'].iloc[-1] > \
    #             data['long_sma'].iloc[-1]:
    #
    #         if not in_position:
    #             print("Signal to BUY!")
    #             order = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=0.01)
    #             in_position = True
    #             print(order)
    #
    #     # If LMA crosses SMA (Going short on the crypto) 👇
    #     elif data['short_sma'].iloc[-2] > data['long_sma'].iloc[-2] and data['short_sma'].iloc[-1] < \
    #             data['long_sma'].iloc[-1]:
    #
    #         if in_position:
    #             print("Signal to SELL!")
    #             order = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=0.01)
    #             in_position = False

    def calculate_rsi(self, close_prices, period=14):
        """
        Calculates the Relative Strength Index (RSI) for a given set of closing prices.
        Args:
            close_prices: A list of closing prices.
            period: The RSI calculation period (default: 14).
        Returns:
            A list of RSI values for each closing price.
        """
        if len(close_prices) < period:
            raise ValueError("Not enough data for RSI calculation.")
        rsi = []
        for i in range(period):
            rsi.append(np.nan)  # Handle initial RSI values
        up_changes = [0] * (period - 1)
        down_changes = [0] * (period - 1)
        for i in range(1, len(close_prices)):
            change = close_prices[i] - close_prices[i - 1]
            if change > 0:
                up_changes.append(change)
                down_changes.append(0)
            else:
                up_changes.append(0)
                down_changes.append(abs(change))
        avg_gain = np.mean(up_changes[period:])
        avg_loss = np.mean(down_changes[period:])
        if avg_loss == 0:
            rsi.extend([100] * (len(close_prices) - period))
        else:
            rsi.extend([100 - (100 / (1 + avg_gain / avg_loss)) for _ in range(len(close_prices) - period)])
        return rsi

if __name__ == '__main__':
    # Apply the strategy to JPM data
    strategy = MovingAverageCrossoverStrategy(short_window=50, long_window=200)
    signals = strategy.generate_signals(data['JPM'])

    # Plot the signals along with the closing price
    # plt.figure(figsize=(14, 7))
    # plt.plot(data['JPM']['Close'], label='JPM Closing Price', alpha=0.5)
    # plt.plot(signals['short_mavg'], label='50-Day SMA', alpha=0.5)
    # plt.plot(signals['long_mavg'], label='200-Day SMA', alpha=0.5)
    # plt.scatter(signals.loc[signals.positions == 1.0].index,
    #             signals.short_mavg[signals.positions == 1.0],
    #             label='Buy Signal', marker='^', color='g', s=100)
    # plt.scatter(signals.loc[signals.positions == -1.0].index,
    #             signals.short_mavg[signals.positions == -1.0],
    #             label='Sell Signal', marker='v', color='r', s=100)
    # plt.title('JPM Moving Average Crossover Strategy')
    # plt.xlabel('Date')
    # plt.ylabel('Price (USD)')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.close()