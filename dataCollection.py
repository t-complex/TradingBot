"""
This file serve as a starting point to download the historical data
from different sources such as yfinance, Coinbase, CryptoCompare,
"""

import yfinance as yf

class DataCollection:

    def yfinanceData(self, cryptos):
        # Download historical data from January 1, 2017, to January 31, 2024
        data = yf.download(cryptos, start="2017-01-01", end="2024-01-31")
        return data

if __name__ == '__main__':
    # Define cryptocurrency symbols
    cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
               "AVAX-USD", "BUSDC-USD", "USDP-USD", "USDT-USD", "USDC-USD", "TUSD-USD"]
    # High Market Cap
    # cryptos = [BTC, ETH, USDT, BNB, SOL, XRP, USDC, ADA, AVAX, DOGE, TRX, DOT, LINK]

    dc = DataCollection()

    # yfinance
    data = dc.yfinanceData(cryptos)
    # Print the first few rows of the data
    print(data.head())
    # Save data to a CSV file
    data.to_csv("data/yfinance_data.csv")