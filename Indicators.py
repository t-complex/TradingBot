"""
Backtesting and Strategy Evaluation
Backtesting is the process of testing a trading strategy on historical data to assess its viability.
We’ll write a backtesting engine in Python and evaluate our moving average crossover strategy.
"""

import ta
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

class Indicators:
    def __init__(self, data, date):
        self.data = data
        self.date = date
    def calculate_indicator(self):
        # Calculate SMA-5-8-13, EMA-13-26, MACD
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_8'] = self.data['Close'].rolling(window=8).mean()
        self.data['SMA_13'] = self.data['Close'].rolling(window=13).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['Histogram'] = self.data['MACD'] - self.data['Signal_Line']
        self.data['MACD_Buy_Signal'] = np.where(self.data['MACD'] > self.data['Signal_Line'], 1, 0)
        self.data['MACD_Sell_Signal'] = np.where(self.data['MACD'] < self.data['Signal_Line'], -1, 0)
        # Calculate Bollinger Bands
        self.data['Mean_BB'] = self.data['Close'].rolling(window=20).mean()
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['Upper_BB'] = self.data['Mean_BB'] + (rolling_std * 2)
        self.data['Lower_BB'] = self.data['Mean_BB'] - (rolling_std * 2)
        self.data['BB_Buy_Signal'] = ((self.data['Close'] < self.data['Lower_BB']) &
                                 (self.data['Close'].shift(1) >= self.data['Lower_BB'].shift(1)))
        self.data['BB_Sell_Signal'] = ((self.data['Close'] > self.data['Upper_BB']) &
                                  (self.data['Close'].shift(1) <= self.data['Upper_BB'].shift(1)))
        # Calculate ATR Indicator
        true_range = np.maximum.reduce([self.data['high'] - self.data['low'],
                                        abs(self.data['high'] - self.data['Close'].shift()),
                                        abs(self.data['low'] - self.data['Close'].shift())])
        self.data['ATR'] = pd.Series(true_range).rolling(window=14).mean()
        self.data['ATR_MA'] = self.data['ATR'].rolling(window=14).mean()
        self.data['ATR_buy_signal'] = ((self.data['ATR'] > self.data['ATR_MA']) &
                                       (self.data['ATR'].shift(1) <= self.data['ATR_MA'].shift(1)))
        self.data['ATR_sell_signal'] = ((self.data['ATR'] < self.data['ATR_MA']) &
                                        (self.data['ATR'].shift(1) >= self.data['ATR_MA'].shift(1)))
        # Calculate RSI Indicator
        delta = self.data['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0], down[down > 0] = 0, 0
        roll_up = up.ewm(span=14).mean()
        roll_down = down.abs().ewm(span=14).mean()
        RS = roll_up / roll_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        self.data['RSI'] = RSI
        self.data['RSI_Buy_Signal'] = np.where(self.data['RSI'] < 30, 1, 0)
        self.data['RSI_Sell_Signal'] = np.where(self.data['RSI'] > 30, -1, 0)
    def calculate_advance_indicators(self):
        self.data['SMA'] = self.data['Close'].rolling(window=20).mean()
        self.data['SD'] = self.data['Close'].rolling(window=20).std()
        self.data['BOLU'] = self.data['SMA_20'] + (self.data['SD_20'] * 2)
        self.data['RSI'] = RSIIndicator(close=self.data['Close'], window=14).rsi()
        self.data['%K'] = StochasticOscillator(self.data['High'], self.data['Low'],
                                               self.data['Close'], window=14, smooth_window=3).stoch()
        self.data['%D'] = StochasticOscillator(self.data['High'], self.data['Low'],
                                               self.data['Close'], window=14, smooth_window=3).stoch_signal()
        self.data['EMA_10'] = EMAIndicator(close=self.data['Close'], window=10).ema_indicator()
        self.data['EMA_20'] = EMAIndicator(close=self.data['Close'], window=20).ema_indicator()
        self.data['ATR_ta'] = AverageTrueRange(self.data['High'], self.data['Low'],
                                            self.data['Close'], window=14).average_true_range()
        self.data['ADX_ta'] = ADXIndicator(self.data['High'], self.data['Low'],
                                        self.data['Close'], window=14).adx()
        normalized_atr = ((ta.atr(60) - ta.lowest(ta.atr(60), 60)) /
                          ((ta.highest(ta.atr(60), 60)) - ta.lowest(ta.atr(60), 60)))
        smoothed_normalized_atr = ta.wma(normalized_atr, 13)
        self.data['ATR_Buy_signal_TA'] = (smoothed_normalized_atr < 0.8 and smoothed_normalized_atr[1] > 0.8 and
                                                            self.data['Close'] > ta.sma(self.data['Close'], 100))
        self.data['ATR_Sell_signal_TA'] = (smoothed_normalized_atr > 0.2 and smoothed_normalized_atr[1] < 0.2 and
                                                            self.data['Close'] < ta.sma(self.data['Close'], 100))
    def visualizing_data(self):
        # Change default background color for all visualizations
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
        fig = go.Figure(layout=layout)
        templated_fig = pio.to_templated(fig)
        pio.templates['my_template'] = templated_fig.layout.template
        pio.templates.default = 'my_template'
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Ohlc(x=self.date,
                              open=self.data['open'],
                              high=self.data['high'],
                              low=self.data['low'],
                              close=self.data['Close'],
                              name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Volume'], name='Volume'), row=2, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.show()
    def visualizing_moving_average_macd(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.date, y=self.data['SMA_20'], name='SMA 20'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['SMA_50'], name='SMA 50'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['EMA_12'], name='EMA 12'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['EMA_26'], name='EMA 26'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Signal_Line'], name='Signal Line'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Histogram'], name='Histogram'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['signal_line_ao'], name='Signal Line AO'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['zero_cross'], name='Zero Cross AO'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_bollinger_bands(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Mean_BB'], name='Mean BB'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Upper_BB'], name='Upper BB'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Lower_BB'], name='Lower BB'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_atr(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.date, y=self.data['ATR'], name='ATR'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['ATR_MA'], name='ATR MA'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_rsi(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.date, y=self.data['RSI'], name='RSI'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_advance_indicators(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.date, y=self.data['SMA'], name='SMA'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['SD'], name='SD'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['EMA_10'], name='EMA 10'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['EMA_12'], name='EMA 12'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['BOLU'], name='BOLU'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['RSI'], name='RSI'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['%K'], name='%K'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['%D'], name='%D'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['ATR_ta'], name='ATR TA'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['ADX_ta'], name='ADX TA'))
        fig.add_trace(go.Scatter(x=self.date, y=self.data['Close'], name='Close', opacity=0.2))
        fig.update_layout(xaxis=dict(type='category'))
        fig.show()
    def visualizing_indicators(self):
        self.visualizing_data()
        self.visualizing_moving_average_macd()
        self.visualizing_bollinger_bands()
        self.visualizing_atr()
        self.visualizing_rsi()