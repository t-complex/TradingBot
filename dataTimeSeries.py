


from darts import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataTimeSeries():
    def __init__(self, data):
        self.data = pd.read_csv("data/all_data.csv")

