import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
import time  
from ib_async import * # type: ignore
import yfinance as yf 
from sklearn.linear_model import LinearRegression

# Calculating sharpe ratio for commodities.
class fetchData():
    def __init__(self, symbol, start, end):
        self.symbol = symbol 
        self.start = start 
        self.end = end 
        self.getdata()
        
    def __repr__(self) -> str:
        return "fetchData(symbol={}, start={}, end={})".format(self.symbol, self.start, self.end)
    
    def getdata(self):
        data = yf.download(self.symbol, self.start, self.end)
        if data.empty: # type: ignore
            raise ValueError("No data downloaded. Check symbol or date range.")
        data = data[['Close']].copy() # type: ignore
        data.rename(columns={"Close": "price"}, inplace=True)
        data["returns"] = np.log(data["price"] / data["price"].shift(1))
        data.dropna(inplace=True)
        self.data = data
        return data

    def calc_sharpe_ratio(self, rfr=0.0175):
        if self.data is None:
            raise ValueError("Data not loaded. Run getdata() before calculating Sharpe ratio.")
        
        annual_return = self.data["returns"].mean() * 252
        annual_volatility = self.data["returns"].std() * np.sqrt(252)
        sharpe = (annual_return - rfr) / annual_volatility
        return sharpe
    def plot_price(self): 
        plt.plot(x = "Date", y = "price")
        plt.show()

# GLD_data = fetchData("GC=F", "2024-10-01", "2025-10-01")
# GLD_data.getdata()
# print(GLD_data.getdata()) 
# print(GLD_data.calc_sharpe_ratio()) 

SIL_data = fetchData("SI=F", "2024-10-01", "2025-10-01")
print(SIL_data.getdata()) 
print(SIL_data.calc_sharpe_ratio()) 
print(SIL_data.plot_price())
