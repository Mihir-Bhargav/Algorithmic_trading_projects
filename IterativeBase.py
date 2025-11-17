
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use("seaborn-v0_8")


# class IterativeBase():
#     ''' Base class for iterative (event-driven) backtesting of trading strategies.
#     '''

#     def __init__(self, symbol, start, end, amount, use_spread = True):
#         '''
#         Parameters
#         ----------
#         symbol: str
#             ticker symbol (instrument) to be backtested
#         start: str
#             start date for data import
#         end: str
#             end date for data import
#         amount: float
#             initial amount to be invested per trade
#         use_spread: boolean (default = True) 
#             whether trading costs (bid-ask spread) are included
#         '''
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
#         ''' Imports the data from detailed.csv (source can be changed).
#         '''
#         raw = pd.read_csv("detailed.csv", parse_dates = ["time"], index_col = "time").dropna()
#         raw = raw.loc[self.start:self.end].copy()
#         raw["returns"] = np.log(raw.price / raw.price.shift(1))
#         self.data = raw

#     def plot_data(self, cols = None):  
#         ''' Plots the closing price for the symbol.
#         '''
#         if cols is None:
#             cols = "price"
#         self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
#     def get_values(self, bar):
#         ''' Returns the date, the price and the spread for the given bar.
#         '''
#         date = str(self.data.index[bar].date())
#         price = round(self.data.price.iloc[bar], 5)
#         spread = round(self.data.spread.iloc[bar], 5)
#         return date, price, spread
    
#     def print_current_balance(self, bar):
#         ''' Prints out the current (cash) balance.
#         '''
#         date, price, spread = self.get_values(bar)
#         print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
#     def buy_instrument(self, bar, units = None, amount = None):
#         ''' Places and executes a buy order (market order).
#         '''
#         date, price, spread = self.get_values(bar)
#         if self.use_spread:
#             price += spread/2 # ask price
#         if amount is not None: # use units if units are passed, otherwise calculate units
#             units = int(amount / price)
#         self.current_balance -= units * price # reduce cash balance by "purchase price"
#         self.units += units
#         self.trades += 1
#         print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
#     def sell_instrument(self, bar, units = None, amount = None):
#         ''' Places and executes a sell order (market order).
#         '''
#         date, price, spread = self.get_values(bar)
#         if self.use_spread:
#             price -= spread/2 # bid price
#         if amount is not None: # use units if units are passed, otherwise calculate units
#             units = int(amount / price)
#         self.current_balance += units * price # increases cash balance by "purchase price"
#         self.units -= units
#         self.trades += 1
#         print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
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
#         ''' Closes out a long or short position (go neutral).
#         '''
#         date, price, spread = self.get_values(bar)
#         print(75 * "-")
#         print("{} | +++ CLOSING FINAL POSITION +++".format(date))
#         self.current_balance += self.units * price # closing final position (works with short and long!)
#         self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
#         print("{} | closing position of {} for {}".format(date, self.units, price))
#         self.units = 0 # setting position to neutral
#         self.trades += 1
#         perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
#         self.print_current_balance(bar)
#         print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
#         print("{} | number of trades executed = {}".format(date, self.trades))
#         print(75 * "-")

class IterativeBase:

    def __init__(self, symbol, start, end, amount, use_spread = True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.get_data()

    def get_data(self):
        raw = yf.download(self.symbol, self.start, self.end)
        raw = raw.copy().dropna() # type: ignore
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={"Close": "price"}, inplace=True)
        raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
        raw["spread"] = 0.005 * (raw["High"] - raw["Low"])
        raw = raw[["price", "spread", "returns"]].dropna().copy()
        self.data = raw
        return raw

    def plot_data(self, cols=None):
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)
        plt.show()

    def get_values(self, bar):
        date = str(self.data.index[bar].date()) # pyright: ignore[reportAttributeAccessIssue]
        price = float(self.data.price.iloc[bar])  # convert to scalar float
        spread = float(self.data.spread.iloc[bar])  # convert to scalar float
        return date, price, spread

    def print_current_balance(self, bar):
        date, price, spread = self.get_values(bar)
        print(f"{date} | Current Balance: {self.current_balance:.2f}")

    def buy_instrument(self, bar, units=None, amount=None):
        date, price, spread = self.get_values(bar)

        if self.use_spread: 
            price += spread/2 #ask price

        if amount is not None:
            units = int(amount / price)
        if units is None:
            raise ValueError("Units must be specified or calculable from amount.")

        self.current_balance -= units * price
        self.units += units
        self.trades += 1
        print(f"{date} | Buying {units} units for {price:.5f}")

    def sell_instrument(self, bar, units=None, amount=None):
        date, price, spread = self.get_values(bar)
        if self.use_spread: 
            price -= spread/2 #bid price

        if amount is not None:
            units = int(amount / price)
        if units is None:
            raise ValueError("Units must be specified or calculable from amount.")

        self.current_balance += units * price
        self.units -= units
        self.trades += 1
        print(f"{date} | Selling {units} units for {price:.5f}")

    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
    def close_pos(self, bar): 
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{}| +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price 
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        print("{}| closing position of {} for {}". format(date, self.units, price))
        self.units = 0 
        self.trades += 1 
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar) 
        print("{}| Net performance (%) = {}".format(date, round(perf, 2)))
        print("{}| Number of trades executed = {}".format(date, self.trades))
        print(75 * "-")
