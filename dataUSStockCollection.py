

import pandas as pd
import yfinance as yf
from datetime import datetime
import quandl

class DataUSStockCollection:

    us_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'QCOM', 'TSLA', 'NVDA']

    # for stock in us_stocks:
