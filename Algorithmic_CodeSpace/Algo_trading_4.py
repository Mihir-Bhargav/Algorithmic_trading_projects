import pandas as pd
import numpy as np 
import time as time 
import math as math 
from ib_async import *  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pytz 
from itertools import product
pd.options.display.float_format = '{:.4f}'.format
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier # added (from sklearn v. 1.7)

# Iterative backtesting => trading based more on events, unlike vectorised backtesting. 

# class IterativeBase:

#     def __init__(self, symbol, start, end, amount, use_spread = True):
#         self.symbol = symbol
#         self.start = start
#         self.end = end
#         self.initial_balance = amount
#         self.current_balance = amount
#         self.units = 0
#         self.trades = 0
#         self.position = 0
#         self.use_spread = use_spread
#         self.get_data()

#     def get_data(self):
#         raw = yf.download(self.symbol, self.start, self.end)
#         raw = raw.copy().dropna() # type: ignore
#         raw = raw.loc[self.start:self.end]
#         raw.rename(columns={"Close": "price"}, inplace=True)
#         raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
#         raw["spread"] = 0.005 * (raw["High"] - raw["Low"])
#         raw = raw[["price", "spread", "returns"]].dropna().copy()
#         self.data = raw
#         return raw

#     def plot_data(self, cols=None):
#         if cols is None:
#             cols = "price"
#         self.data[cols].plot(figsize=(12, 8), title=self.symbol)
#         plt.show()

#     def get_values(self, bar):
#         date = str(self.data.index[bar].date()) # pyright: ignore[reportAttributeAccessIssue]
#         price = float(self.data.price.iloc[bar])  # convert to scalar float
#         spread = float(self.data.spread.iloc[bar])  # convert to scalar float
#         return date, price, spread

#     def print_current_balance(self, bar):
#         date, price, spread = self.get_values(bar)
#         print(f"{date} | Current Balance: {self.current_balance:.2f}")

#     def buy_instrument(self, bar, units=None, amount=None):
#         date, price, spread = self.get_values(bar)

#         if self.use_spread: 
#             price += spread/2 #ask price

#         if amount is not None:
#             units = int(amount / price)
#         if units is None:
#             raise ValueError("Units must be specified or calculable from amount.")

#         self.current_balance -= units * price
#         self.units += units
#         self.trades += 1
#         print(f"{date} | Buying {units} units for {price:.5f}")

#     def sell_instrument(self, bar, units=None, amount=None):
#         date, price, spread = self.get_values(bar)
#         if self.use_spread: 
#             price -= spread/2 #bid price

#         if amount is not None:
#             units = int(amount / price)
#         if units is None:
#             raise ValueError("Units must be specified or calculable from amount.")

#         self.current_balance += units * price
#         self.units -= units
#         self.trades += 1
#         print(f"{date} | Selling {units} units for {price:.5f}")

#     def print_current_position_value(self, bar):
#         ''' Prints out the current position value.
#         '''
#         date, price, spread = self.get_values(bar)
#         cpv = self.units * price
#         print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
#     def print_current_nav(self, bar):
#         ''' Prints out the current net asset value (nav).
#         '''
#         date, price, spread = self.get_values(bar)
#         nav = self.current_balance + self.units * price
#         print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
#     def close_pos(self, bar): 
#         date, price, spread = self.get_values(bar)
#         print(75 * "-")
#         print("{}| +++ CLOSING FINAL POSITION +++".format(date))
#         self.current_balance += self.units * price 
#         self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
#         print("{}| closing position of {} for {}". format(date, self.units, price))
#         self.units = 0 
#         self.trades += 1 
#         perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
#         self.print_current_balance(bar) 
#         print("{}| Net performance (%) = {}".format(date, round(perf, 2)))
#         print("{}| Number of trades executed = {}".format(date, self.trades))
#         print(75 * "-")


# ------------------------
# USAGE EXAMPLE
# ------------------------

# ticker = IterativeBase("BA", "2025-01-01", "2025-10-09", 13500, use_spread=True)

# Optionally print data summary
# print(ticker.data.head())

# # Plot price
# ticker.plot_data()

# # Get first bar values
# print(ticker.get_values(0))

# Buy fixed unit
# ticker.buy_instrument(0, units=50)
# ticker.print_current_balance(0)
# ticker.print_current_position_value(0) #how much the trade is worth. 
# ticker.print_current_nav(0)

# ticker.print_current_balance(-1)
# ticker.print_current_position_value(-1)
# ticker.print_current_nav(-1) 
# ticker.close_pos(-1)


# Building an OOP for IterativeBacktest using the 3 strategies: 
# from IterativeBase import * # pyright: ignore[reportAssignmentType]

# class IterativeBacktest(IterativeBase): 
#     def go_long(self, bar, amount=None, units=None):
#         if self.position == -1: 
#             self.buy_instrument(bar, units, -self.units) #To have even Position before a trade 
#         if units: 
#             self.buy_instrument(bar, units= units)
#         elif amount: 
#             if amount == "all": 
#                 amount = self.current_balance
#                 self.buy_instrument(bar, amount = amount)

#     def go_short(self, bar, amount=None, units = None):
#         if self.position == 1: 
#             self.sell_instrument(bar, units = self.units)
#         if units: 
#             self.sell_instrument(bar, units= self.units)
#         elif amount: 
#             if amount == "all": 
#                 amount = self.current_balance
#                 self.sell_instrument(bar, amount=amount) #Go short

#     def test_sma_strategy(self, SMA_S, SMA_L): 
#         stm = "Testing SMA strategy | {} | SMA_S={} | SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
#         print (75 * "-" ) 
#         print(stm) 
#         print(75* "-") 

#         # reseting: 
#         self.position = 0 
#         self.trades = 0 
#         self.current_balance = self.initial_balance
#         self.get_data()

#         # Defining the strat: 
#         self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean() 
#         self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean() 
#         self.data.dropna(inplace=True) 

#         # Placing order based on the strategy: 
#         for bar in range(len(self.data)-1): 
#             if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: #signal to go long 
#                 if self.position in [0,-1]: 
#                     self.go_long(bar, amount="all") #Go long with all you money, can be changed to any desired amount. 
#                     self.position = 1 

#             elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: 
#                 if self.position in [0, 1]: 
#                     self.go_short(bar, amount='all')
#                     self.position = -1 
#         self.close_pos(bar + 1)



# import IterativeBacktest as IB
# ticker = IterativeBacktest("EURUSD=X", "2010-01-01", "2020-01-01", 10000, use_spread= True)
# ticker.data
# ticker.test_sma_strategy(49, 239)

# Why should the SMA_S and SMA_L be constant? Use ML, AI to make them tweak themselves through data, through trades and optimize themselves based on parameters like rfr, volatility, returns... 


# --------------------------------------------------------------------------------------------------------------------------
# Trading locallt could be risky and unstable; What if your laptop dies, or loses internet connection? You would most likley not close positions as wished and lose money. For this, reason, it would be a good idea to trade using cloud services. 
# --------------------------------------------------------------------------------------------------------------------------

print("Testing after changing the name of the dir")