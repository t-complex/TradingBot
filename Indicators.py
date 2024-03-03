"""
Backtesting and Strategy Evaluation
Backtesting is the process of testing a trading strategy on historical data to assess its viability.
We’ll write a backtesting engine in Python and evaluate our moving average crossover strategy.
"""

import ta
import numpy as np
import pandas as pd
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class Indicators:
    def calculate_moving_averages_macd(self, data):
        data['SMA_5'] = data['close'].rolling(window=5).mean()
        data['SMA_8'] = data['close'].rolling(window=8).mean()
        data['SMA_13'] = data['close'].rolling(window=13).mean()
        data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['Histogram'] = data['MACD'] - data['Signal_Line']
        data['MACD_Buy_Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, 0)
        data['MACD_Sell_Signal'] = np.where(data['MACD'] < data['Signal_Line'], -1, 0)
        return data
    def calculate_bollinger_bands(self, data, window_length=20, num_std=2):
        data['Mean_BB'] = data['close'].rolling(window=window_length).mean()
        rolling_std = data['close'].rolling(window=window_length).std()
        data['Upper_BB'] = data['Mean_BB'] + (rolling_std * num_std)
        data['Lower_BB'] = data['Mean_BB'] - (rolling_std * num_std)
        data['BB_Buy_Signal'] = (data['close'] < data['Lower_BB']) & (data['close'].shift(1) >= data['Lower_BB'].shift(1))
        data['BB_Sell_Signal'] = (data['close'] > data['Upper_BB']) & (data['close'].shift(1) <= data['Upper_BB'].shift(1))
        return data
    def calculate_atr(self, data):
        true_range = np.maximum.reduce([data['high'] - data['low'], abs(data['high'] - data['close'].shift()), abs(data['low'] - data['close'].shift())])
        data['ATR'] = pd.Series(true_range).rolling(window=14).mean()
        data['ATR_MA'] = data['ATR'].rolling(window=14).mean()
        data['ATR_buy_signal'] = (data['ATR'] > data['ATR_MA']) & (data['ATR'].shift(1) <= data['ATR_MA'].shift(1))
        data['ATR_sell_signal'] = (data['ATR'] < data['ATR_MA']) & (data['ATR'].shift(1) >= data['ATR_MA'].shift(1))
        return data
    def calculate_rsi(self, data, window_length=14, overbought=70, oversold=30):
        delta = data['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0], down[down > 0] = 0, 0
        roll_up = up.ewm(span=window_length).mean()
        roll_down = down.abs().ewm(span=window_length).mean()
        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        data['RSI'] = RSI
        data['RSI_Buy_Signal'] = np.where(data['RSI'] < oversold, 1, 0)
        data['RSI_Sell_Signal'] = np.where(data['RSI'] > overbought, -1, 0)
        return data

    def visualizing_data(self, data, date):
        # Change default background color for all visualizations
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
        fig = go.Figure(layout=layout)
        templated_fig = pio.to_templated(fig)
        pio.templates['my_template'] = templated_fig.layout.template
        pio.templates.default = 'my_template'
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Ohlc(x=date,
                              open=data['open'],
                              high=data['high'],
                              low=data['low'],
                              close=data['close'],
                              name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=data['Volume'], name='Volume'), row=2, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()
    def visualizing_moving_average_macd(self, data, date):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date, y=data['SMA_20'], name='SMA 20'))
        fig.add_trace(go.Scatter(x=date, y=data['SMA_50'], name='SMA 50'))
        fig.add_trace(go.Scatter(x=date, y=data['EMA_12'], name='EMA 12'))
        fig.add_trace(go.Scatter(x=date, y=data['EMA_26'], name='EMA 26'))
        fig.add_trace(go.Scatter(x=date, y=data['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=date, y=data['Signal_Line'], name='Signal Line'))
        fig.add_trace(go.Scatter(x=date, y=data['Histogram'], name='Histogram'))
        fig.add_trace(go.Scatter(x=date, y=data['signal_line_ao'], name='Signal Line AO'))
        fig.add_trace(go.Scatter(x=date, y=data['zero_cross'], name='Zero Cross AO'))
        fig.add_trace(go.Scatter(x=date, y=data['close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_bollinger_bands(self, data, date):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date, y=data['Mean_BB'], name='Mean BB'))
        fig.add_trace(go.Scatter(x=date, y=data['Upper_BB'], name='Upper BB'))
        fig.add_trace(go.Scatter(x=date, y=data['Lower_BB'], name='Lower BB'))
        fig.add_trace(go.Scatter(x=date, y=data['close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_atr(self, data, date):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date, y=data['ATR'], name='ATR'))
        fig.add_trace(go.Scatter(x=date, y=data['ATR_MA'], name='ATR MA'))
        fig.add_trace(go.Scatter(x=date, y=data['close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()