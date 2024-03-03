"""
This file serve as a starting point to download the historical data
from different sources such as yfinance, Coinbase, CryptoCompare,
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

class DataCollection:
    def dataCryptoCollection(self):
        # Define cryptocurrency symbols
        # cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
        #            "AVAX-USD", "BUSDC-USD", "USDP-USD", "USDT-USD", "USDC-USD", "TUSD-USD"]
        cryptos = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'ADA', 'DOGE']
        today_date = datetime.today().date()
        data = pd.DataFrame()
        for crypto in cryptos:
            all_price_data = yf.download(f'{crypto}-USD', interval='1d', period='10y')
            if crypto == 'BTC':
                data.index = all_price_data.index
                # add missing dates in index
                idx_dates = list(pd.date_range(min(data.index), today_date))
                data = data.reindex(idx_dates, method='ffill')
            data[f'{crypto}'] = all_price_data['Close']
            data.to_csv(f'{crypto}.csv')
    def dataUSStockCollection(self):
        us_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'QCOM', 'TSLA', 'NVDA']
        today_date = datetime.today().date()