"""
This file serve as a starting point to download the historical data
from different sources such as yfinance, Coinbase, CryptoCompare,
"""
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataCryptoCollection:

    def yfinanceData(self, cryptos, todays_date):
        data = {}
        for crypto in cryptos:
            data[crypto] = yf.download(f'{crypto}-USD', end=todays_date)
        return data

if __name__ == '__main__':
    # Define cryptocurrency symbols
    # cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    #            "AVAX-USD", "BUSDC-USD", "USDP-USD", "USDT-USD", "USDC-USD", "TUSD-USD"]
    cryptos = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'ADA', 'DOGE']

    today_date = datetime.today().date()

    dc = DataCryptoCollection()
    data = dc.yfinanceData(cryptos, today_date)
    # Save data to a CSV file
    data.to_csv("data/all_data.csv")