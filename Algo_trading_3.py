# This will be used for the thrid part of the course, implementing, testing statergies. 
import pandas as pd
import numpy as np 
import time as time 
import math as math 
from ib_async import *  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pytz 
from itertools import product
pd.options.display.float_format = '{:.4f}'.format
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier # added (from sklearn v. 1.7)



# start = "2020-01-01"
# end = "2025-06-01" 
# symbol = "EURUSD=X"
# df = yf.download(symbol, start, end)
# print(df.Close)
# df.Close.to_csv("eurusd.csv")


# Creating a SMA tradign statergy>>
# data = pd.read_csv("eurusd.csv", parse_dates=["Date"], index_col = "Date")
# sma_s = 50
# sma_l = 200

# data["SMA_S"] = data.Price.rolling(sma_s).mean()
# data["SMA_L"] = data.Price.rolling(sma_l).mean()
# # data.plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
# # plt.legend(fontsize = 12)
# # plt.show()

# data.dropna(inplace = True)
# data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1 )
# data["returns"] = np.log(data.Price.div(data.Price.shift(1)))
# data["strategy"] = data.position.shift(1) * data["returns"]
# data.dropna(inplace = True)
# # print(data[["returns", "strategy"]].sum())
# # print(data[["returns", "strategy"]].sum()) # absolute performance
# # print(data[["returns", "strategy"]].mean() * 252) # annualized return
# # print(data[["returns", "strategy"]].std() * np.sqrt(252))
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
# # print(data[["creturns"]])
# # print(data[["cstrategy"]])
# data[["creturns", "cstrategy"]].plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
# plt.legend(fontsize = 12)
# plt.show()
# print(data)

# df = pd.read_csv("eurusd.csv", parse_dates = ["Date"], index_col = "Date")
# # print(df)

# def test_strategy(SMA):
#     data = df.copy() # type: ignore
#     data["returns"] = np.log(data.Price.div(data.Price.shift(1)))
#     data["SMA_S"] = data.Price.rolling(int(SMA[0])).mean()
#     data["SMA_L"] = data.Price.rolling(int(SMA[1])).mean()
#     data.dropna(inplace = True)
    
#     data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
#     data["strategy"] = data.position.shift(1) * data["returns"]
#     data.dropna(inplace = True)
#     return np.exp(data["strategy"].sum())

# # print(test_strategy((20, 200)))

# SMA_S_range = range(10, 50, 1)
# SMA_L_range = range(100, 252, 1)
# list(product(SMA_S_range, SMA_L_range))
# combinations = list(product(SMA_S_range, SMA_L_range))
# # print(len(combinations))

# results = []
# for comb in combinations:
#     results.append(test_strategy(comb))

# print(np.max(results))
# print(np.argmax(results))
# print(combinations[np.argmax(results)])
# many_results = pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
# many_results["performance"] = results 
# print(many_results)
# We found a statergy, optimized, ran tests and found that the leading algorithsm was a SMA with (30, 103) and delivered 25% returns. Pretty Cool!!!

# class SMABacktester():
#     def __init__ (self, symbol, SMA_S, SMA_L, start, end):
#         self.symbol = symbol 
#         self.SMA_S = SMA_S
#         self.SMA_L = SMA_L
#         self.start = start 
#         self.end = end
#         self.get_data()
#         self.prepare_data()
    
#     def __repr__(self):
#         return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {}, start = {}, end = {})".format(self.symbol, self.SMA_S, self.SMA_L, self.start, self.end)
    
#     def get_data(self):
#         ''' Imports the data from forex_pairs.csv (source can be changed).
#         '''
#         raw = pd.read_csv("forex_pairs.csv", parse_dates = ["Date"], index_col = "Date")
#         raw = raw[self.symbol].to_frame().dropna()
#         raw = raw.loc[self.start:self.end].copy()
#         raw.rename(columns={self.symbol: "price"}, inplace=True)
#         raw["returns"] = np.log(raw / raw.shift(1))
#         self.data = raw
        
#     def prepare_data(self):
#         '''Prepares the data for strategy backtesting (strategy-specific).
#         '''
#         data = self.data.copy()
#         data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
#         data["SMA_L"] = data["price"].rolling(self.SMA_L).mean()
#         self.data = data

#     def set_parameters(self, SMA_S = None, SMA_L = None):
#         if SMA_S is not None:
#             self.SMA_S = SMA_S
#             self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
#         if SMA_L is not None:
#             self.SMA_L = SMA_L
#             self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
#     def test_strategy(self):
#         data = self.data.copy().dropna()
#         data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
#         data["strategy"] = data["position"].shift(1) * data["returns"]
#         data.dropna(inplace=True)
#         data["creturns"] = data["returns"].cumsum().apply(np.exp)
#         data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
#         self.results = data
#         perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
#         outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
#         return round(perf, 6), round(outperf, 6)
#     def plot_results(self):
#         if self.results is None:
#             print("Run test_strategy() first.")
#         else:
#             title = "{} | SMA_S = {} | SMA_L = {}".format(self.symbol, self.SMA_S, self.SMA_L)
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

        
#     def optimize_parameters(self, SMA_S_range, SMA_L_range):
    
#         combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
        
#         # test all combinations
#         results = []
#         for comb in combinations:
#             self.set_parameters(comb[0], comb[1])
#             results.append(self.test_strategy()[0])
        
#         best_perf = np.max(results) # best performance
#         opt = combinations[np.argmax(results)] # optimal parameters
        
#         # run/set the optimal strategy
#         self.set_parameters(opt[0], opt[1])
#         self.test_strategy()
                   
#         # create a df with many results
#         many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
#         many_results["performance"] = results
#         self.results_overview = many_results
#         return opt, best_perf
    
# tester = SMABacktester("EURUSD=X", 50, 200, "2010-01-01", "2020-01-01")
# print(tester.optimize_parameters((10, 50, 1), (100, 252, 1)))



# data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time") 
# data["returns"] = np.log(data.div(data.shift(1)))
# to_plot = ["returns"]
# data.dropna(inplace = True)

# window = 3
# print(data["returns"].rolling(window).mean())
# data["position"] = -np.sign(data["returns"].rolling(window).mean()) # contrarian (minus sign), else it is just a momentum staratergy. 
# data["strategy"] = data.position.shift(1) * data["returns"] 
# data.dropna(inplace = True)
# print(data.dropna(inplace = True))
# print(data[["returns", "strategy"]].sum().apply(np.exp))
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
# data[["creturns", "cstrategy"]].plot(figsize = (12 , 8), title = "EUR/USD | Window = {}".format(window), fontsize = 12)
# plt.show()


# for w in [1, 2, 3, 5, 10]:
#     data["position{}".format(w)] = -np.sign(data["returns"].rolling(w).mean())
#     data["strategy{}".format(w)] = data["position{}".format(w)].shift(1) * data["returns"]
#     to_plot.append("strategy{}".format(w))

# print(data)
# data[to_plot].dropna().cumsum().apply(np.exp).plot(figsize = (12, 8))
# plt.title("DJI Intraday - 6h bars", fontsize = 12)
# plt.legend(fontsize = 12)
# plt.show() 

# Working with trading costs, very important aspect when it come to P&L. 

# data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time")
# window = 3
# data["returns"] = np.log(data.div(data.shift(1)))
# data["position"] = -np.sign(data["returns"].rolling(window).mean())
# data["strategy"] = data.position.shift(1) * data["returns"]
# # print(data)
# data.dropna(inplace = True)
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

# data.loc[:, "position"].plot(figsize = (12 , 8))
# # plt.show()

# spread = 1.5 * 0.0001 # pips == fourth price decimal
# half_spread = spread / 2 # absolute tc per trade (position change +-1)
# ptc = half_spread / data.price.mean() # proportional tc per trade (position change +-1) 
# # print(ptc)
# # print(data.position.diff().fillna(0).abs())
# data["trades"] = data.position.diff().fillna(0).abs()
# # print(data)
# # print(data.trades.value_counts())

# data["strategy_net"] = data.strategy - data.trades * ptc
# data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)
# data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))
# plt.show()

# class ConBacktester():
#     def __init__(self, symbol, start, end):
#         self.symbol = symbol
#         self.start = start 
#         self.end = end
#         self.results = None
#         self.data = None  # Initialize data attribute

#     def __repr__(self):
#         return f"ConBacktester(symbol={self.symbol}, start={self.start}, end={self.end})"

#     def getdata(self):
#         data = yf.download(self.symbol, self.start, self.end)
#         if data.empty: # type: ignore
#             raise ValueError("No data downloaded. Check symbol or date range.")
#         data = data[['Close']].copy() # type: ignore
#         data.rename(columns={"Close": "price"}, inplace=True)
#         data["returns"] = np.log(data["price"] / data["price"].shift(1))
#         self.data = data

#     def calcData(self, window=1):
#         if self.data is None:
#             raise AttributeError("Data not loaded. Run getdata() first.")

#         self.window = window
#         roll_mean = self.data["returns"].rolling(window).mean()
#         self.data["position"] = -np.sign(roll_mean)
#         self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
#         self.data.dropna(inplace=True)
#         self.data["creturns"] = np.exp(self.data["returns"].cumsum())
#         self.data["cstrategy"] = np.exp(self.data["strategy"].cumsum())
#         self.results = self.data.copy()

#         perf = self.data["cstrategy"].iloc[-1]
#         outperf = perf - self.data["creturns"].iloc[-1]
#         return round(perf, 6), round(outperf, 6)

#     def plot_result(self):
#         if self.results is None:
#             print("Run calcData() first.")
#         else:
#             title = f"{self.symbol} | Window = {self.window} | TC = {getattr(self, 'tc', 'N/A')}"
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
#             plt.show()

#     def optimize_parameters(self, windows_range):
#         windows = range(*windows_range)
#         results = []

#         for window in windows:
#             try:
#                 perf, _ = self.calcData(window)
#                 results.append(perf)
#             except Exception as e:
#                 print(f"Window {window} failed: {e}")
#                 results.append(np.nan)

#         results = np.array(results)
#         valid_idx = ~np.isnan(results)

#         if not valid_idx.any():
#             raise ValueError("No valid results during optimization.")

#         best_perf = np.max(results[valid_idx])
#         opt = np.array(list(windows))[valid_idx][np.argmax(results[valid_idx])]

#         self.calcData(opt)
#         self.results_overview = pd.DataFrame({"window": list(windows), "performance": results})
#         return opt, round(best_perf, 6)
    
# tester = ConBacktester("EURUSD=X", "2018-01-01", "2019-12-31")
# tester.getdata()
# print("Data head:\n", tester.data.head()) # type: ignore
# perf, outperf = tester.calcData(window=5)
# print("Performance:", perf)
# print("Outperformance:", outperf)
# tester.plot_result()
# opt_window, best_perf = tester.optimize_parameters((2, 10))
# print("Best window:", opt_window)
# print("Best performance:", best_perf)
# print("Optimization results:\n", tester.results_overview.head())

# data = pd.read_csv("intraday.csv", parse_dates = ["time"], index_col = "time") # type: ignore
# data["returns"] = np.log(data.div(data.shift(1))) # type: ignore
# SMA = 30
# dev = 2
# data["SMA"] = data["price"].rolling(SMA).mean()
# # data[["price", "SMA"]].plot(figsize = (12, 8))
# # data.loc["2019-08", ["price", "SMA"]].plot(figsize = (12, 8)) # type: ignore
# data["price"].rolling(SMA).std()
# data["price"].rolling(SMA).std().plot(figsize = (12, 8))
# # As bollinger bands have +- x(std.v), we can define lower and upper bands by:
# data["Lower"] = data["SMA"] - data["price"].rolling(SMA).std() * dev # Lower Band -2 Std Dev
# data["Upper"] = data["SMA"] + data["price"].rolling(SMA).std() * dev # Upper Band -2 Std Dev
# data.drop(columns = "returns").plot(figsize = (12, 8))
# # plt.show()
# # This shows the lower and upper bounds.
# data["distance"] = data.price - data.SMA # helper Column
# data["position"] = np.where(data.price < data.Lower, 1, np.nan) # 1. oversold -> go long 
# data["position"] = np.where(data.price > data.Upper, -1, data["position"]) # 2. overbought -> go short 
# # 3. crossing SMA ("Middle Band") -> go neutral
# data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
# data["position"] = data.position.ffill().fillna(0) # where 1-3 isn´t applicable -> hold previous position
# # print(data)
# # print(data.position.value_counts())
# data.drop(columns = ["returns", "distance"]).loc["2019-08"].plot(figsize = (12, 8), secondary_y = "position") # type: ignore
# data.position.plot(figsize = (12, 8))
# # plt.show()
# data["strategy"] = data.position.shift(1) * data["returns"]
# data.dropna(inplace = True)
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
# data[["creturns", "cstrategy"]].plot(figsize = (12 , 8))
# ptc = 0.00007 
# data["trades"] = data.position.diff().fillna(0).abs() 
# print(data.trades.value_counts())
# data["strategy_net"] = data.strategy - data.trades * ptc
# data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)
# print(data)
# data[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12 , 8))
# # plt.show()
# # Look at figure_20 in charts. The difference between red and green,net profit and profit is miniaml which is a sign of an efficient trading algorithm. More importantly, overtrading can and leads to losing profits via commission. 


# class MeanRevBacktester():
#     ''' Class for the vectorized backtesting of Bollinger Bands-based trading strategies.
#     '''
    
#     def __init__(self, symbol, SMA, dev, start, end, tc):
#         '''
#         Parameters
#         ----------
#         symbol: str
#             ticker symbol (instrument) to be backtested
#         SMA: int
#             moving window in bars (e.g. days) for SMA
#         dev: int
#             distance for Lower/Upper Bands in Standard Deviation units
#         start: str
#             start date for data import
#         end: str
#             end date for data import
#         tc: float
#             proportional transaction/trading costs per trade
#         '''
#         self.symbol = symbol
#         self.SMA = SMA
#         self.dev = dev
#         self.start = start
#         self.end = end
#         self.tc = tc
#         self.results = None
#         self.get_data()
#         self.prepare_data()
        
#     def __repr__(self):
#         rep = "MeanRevBacktester(symbol = {}, SMA = {}, dev = {}, start = {}, end = {})"
#         return rep.format(self.symbol, self.SMA, self.dev, self.start, self.end)
        
#     def get_data(self):
#         ''' Imports the data from intraday_pairs.csv (source can be changed).
#         '''
#         raw = pd.read_csv("intraday_pairs.csv", parse_dates = ["time"], index_col = "time")
#         raw = raw[self.symbol].to_frame().dropna()
#         raw = raw.loc[self.start:self.end]
#         raw.rename(columns={self.symbol: "price"}, inplace=True)
#         raw["returns"] = np.log(raw / raw.shift(1))
#         self.data = raw
        
#     def prepare_data(self):
#         '''Prepares the data for strategy backtesting (strategy-specific).
#         '''
#         data = self.data.copy()
#         data["SMA"] = data["price"].rolling(self.SMA).mean()
#         data["Lower"] = data["SMA"] - data["price"].rolling(self.SMA).std() * self.dev
#         data["Upper"] = data["SMA"] + data["price"].rolling(self.SMA).std() * self.dev
#         self.data = data
        
#     def set_parameters(self, SMA = None, dev = None):
#         ''' Updates parameters (SMA, dev) and the prepared dataset.
#         '''
#         if SMA is not None:
#             self.SMA = SMA
#             self.data["SMA"] = self.data["price"].rolling(self.SMA).mean()
#             self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
#             self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
            
#         if dev is not None:
#             self.dev = dev
#             self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
#             self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
            
#     def test_strategy(self):
#         ''' Backtests the Bollinger Bands-based trading strategy.
#         '''
#         data = self.data.copy().dropna()
#         data["distance"] = data.price - data.SMA
#         data["position"] = np.where(data.price < data.Lower, 1, np.nan)
#         data["position"] = np.where(data.price > data.Upper, -1, data["position"])
#         data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
#         data["position"] = data.position.ffill().fillna(0)
#         data["strategy"] = data.position.shift(1) * data["returns"]
#         data.dropna(inplace = True)
        
#         # determine the number of trades in each bar
#         data["trades"] = data.position.diff().fillna(0).abs()
        
#         # subtract transaction/trading costs from pre-cost return
#         data.strategy = data.strategy - data.trades * self.tc
        
#         data["creturns"] = data["returns"].cumsum().apply(np.exp)
#         data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
#         self.results = data
       
#         perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
#         outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
#         return round(perf, 6), round(outperf, 6)
    
#     def plot_results(self):
#         ''' Plots the performance of the trading strategy and compares to "buy and hold".
#         '''
#         if self.results is None:
#             print("Run test_strategy() first.")
#         else:
#             title = "{} | SMA = {} | dev = {} | TC = {}".format(self.symbol, self.SMA, self.dev, self.tc)
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))     
   
#     def optimize_parameters(self, SMA_range, dev_range):
#         ''' Finds the optimal strategy (global maximum) given the Bollinger Bands parameter ranges.

#         Parameters
#         ----------
#         SMA_range, dev_range: tuple
#             tuples of the form (start, end, step size)
#         '''
        
#         combinations = list(product(range(*SMA_range), range(*dev_range)))
        
#         # test all combinations
#         results = []
#         for comb in combinations:
#             self.set_parameters(comb[0], comb[1])
#             results.append(self.test_strategy()[0])
        
#         best_perf = np.max(results) # best performance
#         opt = combinations[np.argmax(results)] # optimal parameters
        
#         # run/set the optimal strategy
#         self.set_parameters(opt[0], opt[1])
#         self.test_strategy()
                   
#         # create a df with many results
#         many_results =  pd.DataFrame(data = combinations, columns = ["SMA", "dev"])
#         many_results["performance"] = results
#         self.results_overview = many_results
                            
#         return opt, best_perf
    
# tester = MeanRevBacktester("GBPUSD", 30, 2, "2018-01-01", "2019-12-31", 0) 

# print(tester.test_strategy())
# print(tester.plot_results())
# print(tester.optimize_parameters((25, 100, 1), (1, 5, 1)))
# print(tester.plot_results())
# plt.show()

# class ConBacktester(): 
#     def __init__ (self, start, end, symbol, tc):
#         self.start = start 
#         self.end = end 
#         self.symbol = symbol 
#         self.tc = tc 
#     def __repr__(self):
#         return "ConBacktester(start={}, end={}, symbol={})".format(self.start, self.end, self.symbol)
#     def get_data(self): 
#         data = yf.download(self.symbol, self.start, self.end)
#         if data is None or data.empty:
#             raise ValueError("No data downloaded. Make sure that the start, end, symbol is inputted correctly.")
#         data = data[["Close"]].copy()
#         data = data.rename(columns={"Close": "price"})
#         data["returns"] = np.log(data["price"] / data["price"].shift(1))
#         self.data = data
#     def test_strategy(self, window = 1):
#         self.window = window 
#         data = self.data.copy().dropna()
#         data["position"] = -np.sign(data["returns"].rolling(self.window).mean()) 
#         data["strategy"] = data["position"].shift(1) * data["returns"]
#         data.dropna(inplace=True) 
#         data["trades"] = data.position.diff().fillna(0).abs() 
#         data["cstrategy"] =  data["strategy"].cumsum().apply(np.exp) 
#         data["creturns"] = data["returns"].cumsum().apply(np.exp) 
#         data.strategy = data.strategy - data.trades * self.tc 
#         self.results = data 
#         perf = data["cstrategy"].iloc[-1]
#         outperf = perf  - data['creturns'].iloc[-1]
#         return round (perf ,6), (outperf, 6) 
#     def plot_results(self):
#         if self.results is None:
#             print("Please run test_strategy() first: ")
#         else:
#             title = "{} | Window = {} | TC={}".format(self.symbol, self.window, self.tc)
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12,8) )
#     def optimize_parameter(self, window_range): 
#         windows = range(*window_range)
#         results = []
#         for window in windows: 
#             results.append(self.test_strategy(window)[0]) # pyright: ignore[reportUndefinedVariable]
#         best_performance = np.max(results)
#         opt = windows[np.argmax(results)]
#         self.test_strategy(opt)
#         many_results = pd.DataFrame(data = {"window" : windows, "performance": results})
#         self.results_oveview = many_results
#         return opt, best_performance
    
# # Step 1: Create the backtester
# bt = ConBacktester(start="2022-01-01", end="2022-12-31", symbol="AAPL", tc=0.001)
# print(bt)  # Check representation

# # Step 2: Download and prepare data
# bt.get_data()
# print(bt.data.head())  # Preview data

# # Step 3: Run strategy with a specific window
# perf, outperf = bt.test_strategy(window=5)
# print("Performance:", perf)
# print("Outperformance:", outperf)

# # Step 4: Plot strategy vs benchmark
# bt.plot_results()
# plt.show()

# # Step 5: Optimize over a range of windows
# opt_window, best_perf = bt.optimize_parameter(window_range=(2, 10))
# print("Best window:", opt_window)
# print("Best performance:", round(best_perf, 6))

# # Step 6: Show optimization results
# print(bt.results_oveview.head())

# class SMABacktester(): 
#     def __init__ (self, start, end, symbol, SMA_S, SMA_L): 
#         self.start = start 
#         self.end = end 
#         self.symbol = symbol 
#         self.SMA_S = SMA_S
#         self.SMA_L = SMA_L
#     def __repr__(self): 
#         return "SMABacktester(start = {}, end = {}, symbol= {}, SMA_S = {}, SMA_L ={})".format(self.start, self.end, self.symbol, self.SMA_S, self.SMA_L) 
#     def get_data(self): 
#         raw = yf.download(self.symbol, self.start, self.end)
#         raw = raw.copy().dropna() # type: ignore
#         raw = raw.loc[self.start:self.end]
#         raw.rename(columns={"Close": "price"}, inplace=True)
#         raw["returns"] = np.log(raw["price"]/ raw["price"].shift(1))
#         self.data = raw
#         return raw 
#     def prepare_data(self): 
#         data = self.data.copy() 
#         data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
#         data["SMA_L"] = data["price"].rolling(self.SMA_L).mean() 
#         self.raw = data
#         return data 
#     def set_parameters(self, SMA_S = None, SMA_L = None): 
#         if SMA_S is not None:
#             self.SMA_S = SMA_S
#             self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
#         if SMA_L is not None:
#             self.SMA_L = SMA_L
#             self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
#     def test_strategy(self, window = 1): 
#         data = self.raw 
#         data["position"] = np.where(data["SMA_L"] > data["SMA_S"], 1, -1) 
#         data.dropna(inplace=True)
#         data["creturns"] = data["returns"].cumsum().apply(np.exp)
#         data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
#         perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
#         outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
#         self.results = data
#         return round(perf, 6), round(outperf, 6)

#     def plot_results(self, results): 
#         if self.results is None: 
#             print("Please run the test_strategy() first:") 
#         else: 
#             title = "{} | SMA_S = {} | SMA_L = {}". format(self.symbol, self.SMA_S, self.SMA_L)
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
#     def optimize_parameters(self, SMA_S_range, SMA_L_range): 
#         results = []
#         combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
#         for comb in combinations: 
#             self.set_parameters(comb[0], comb[1])
#             results.append(self.test_strategy()[0])
        
#         perf = np.max(results) 
#         opt = combinations[np.argmax(results)] 
#         self.set_parameters(opt[0], opt[1])
#         self.test_strategy() 

#         many_results = pd.DataFrame( data = combinations, columns=["SMA_S", "SMA_L"] )
#         many_results['performance'] = results 
#         self.results_overview = many_results               
#         return opt, perf



# # ML!!!!!!
# budget = np.array([5, 10, 17, 27, 35, 40, 42, 49, 54, 60])
# revenue = np.array([2.6, 19. , 23.8, 26.9, 41.1, 58.3, 40.3, 58.7, 73.1, 69.7])
# revenue_new = np.array([74.2,  80.7, 98.2,  94.8, 101.7]) 
# budget_new = np.array([63, 66, 74, 80, 85]) 
# df = pd.DataFrame(data = {"revenue":revenue, "budget":budget})
# df_new = pd.DataFrame(data = {"revenue":revenue_new, "budget":budget_new})

# lm = LinearRegression(fit_intercept = True)
# lm.fit(X = df.budget.to_frame(), y = df.revenue) # fitting the model (Regression Line)
# slope = lm.coef_ # slope of Regression Line
# intercept = lm.intercept_ # intercept of Regression Line
# df["pred"] = lm.predict(df.budget.to_frame())
# x_lin = np.array([0, 100])
# y_lin = intercept + slope * x_lin

# plt.figure( figsize = (12, 8))
# plt.scatter(x = df.budget, y = df.revenue, s = 50, label = "Data")
# plt.plot(x_lin, y_lin, c = "red", label = "Regression Line")
# plt.xlabel("Budget", fontsize = 13)
# plt.ylabel("Revenue", fontsize = 13)
# plt.legend(fontsize = 13)
# plt.show()

# poly_m = np.polyfit(x= df.budget, y=df.revenue, deg = 9)
# x_poly = np.linspace(0, 100, 1000) # x values for polynomial regression line/curve
# y_poly = np.polyval(poly_m, x_poly) # y values for polynomial regression line/curve
# plt.figure( figsize = (12, 8))
# plt.scatter(x = df.budget, y = df.revenue, s = 50, label = "Data")
# plt.plot(x_lin, y_lin, c = "red", label = "Linear Regression Line")
# plt.plot(x_poly, y_poly, label = "Polynomial Regression | deg = 9 (Overfit)",linestyle = "--", color = "red")
# plt.scatter(x = df_new.budget, y = df_new.revenue, s = 50, label = "New Data")
# plt.xlabel("Budget", fontsize = 13)
# plt.ylabel("Revenue", fontsize = 13)
# plt.legend(fontsize = 11, loc = 4)
# plt.ylim(0, 150)
# plt.show()

# data = pd.read_csv("five_minute.csv", parse_dates = ["time"], index_col = "time")
# data.plot(figsize = (12, 8))
# # plt.show()
# data["returns"] = np.log(data/ data.shift(1))
# data["lag1"] = data.returns.shift(1)
# data.dropna(inplace=True)

# lm = LinearRegression(fit_intercept=True) 
# lm.fit(data.lag1.to_frame(), data.returns)
# slope = lm.coef_
# intercept = lm.intercept_
# print(slope, intercept)
# data["pred"] = lm.predict(data.lag1.to_frame())

# plt.figure(figsize = (12, 8))
# plt.scatter(x = data.lag1, y = data.returns, label = "Data")
# plt.plot(data.lag1, data.pred, c = "red", label = "Linear Regression")
# plt.xlim(-0.005, 0.005)
# plt.ylim(-0.005, 0.005)
# plt.legend(fontsize = 13)
# plt.xlabel("Lag1 Returns", fontsize = 13)
# plt.ylabel("Returns", fontsize = 13)
# plt.show()

# data.pred = np.sign(data.pred) # maybe the model can predict the market direction (-1 / +1)
# print(data)
# hits = np.sign(data.returns * data.pred).value_counts() # type: ignore
# print(hits)
# hit_ratio = hits[1.0] / sum(hits)
# print(hit_ratio)


# data = pd.read_csv("five_minute.csv", parse_dates = ["time"], index_col = "time")
# data.dropna(inplace = True)
# data["returns"] = np.log(data.div(data.shift(1)))
# lags = 5
# cols = []
# for lag in range(1, lags + 1):
#     col = "lag{}".format(lag)
#     data[col] = data.returns.shift(lag)
#     cols.append(col)
# data.dropna(inplace = True)
# lm = LinearRegression(fit_intercept = True)
# lm.fit(data[cols], data.returns) 
# print(lm.coef_) 
# print(lm.intercept_)
# data["pred"] = lm.predict(data[cols])
# data.pred = np.sign(data.pred)
# data.pred.value_counts() # type: ignore
# hits = np.sign(data.returns * data.pred).value_counts() # type: ignore
# hit_ratio = hits[1.0] / sum(hits)
# print(hit_ratio)
# data["strategy"] = data.pred * data.returns
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
# data[["creturns", "cstrategy"]].plot(figsize = (12 , 8))
# data["trades"] = data.pred.diff().fillna(0).abs() # pyright: ignore[reportAttributeAccessIssue]
# plt.show()
# print(data.trades.value_counts())
# print(data)

# data = pd.read_csv("test_set.csv", parse_dates = ["time"], index_col = "time")
# data["returns"] = np.log(data.div(data.shift(1)))
# lm = LinearRegression(fit_intercept = True)
# lags = 5
# cols = []
# for lag in range(1, lags + 1):
#     col = "lag{}".format(lag)
#     data[col] = data.returns.shift(lag)
#     cols.append(col)
# data.dropna(inplace = True)
# lm.fit(data[cols], data.returns)
# data["pred"] = lm.predict(data[cols])
# data["pred"] = np.sign(data["pred"])
# data["pred"].value_counts()  # This is now a pandas Series
# hits = np.sign(data["returns"] * data["pred"])
# hits_counts = hits.value_counts() # pyright: ignore[reportAttributeAccessIssue]
# hit_ratio = hits_counts[1.0] / hits_counts.sum()
# print(hit_ratio)
# data["strategy"] = data["pred"] * data["returns"]
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
# data[["creturns", "cstrategy"]].plot(figsize = (12 , 8))
# plt.show()
# data["trades"] = data["pred"].diff().fillna(0).abs()
# print(data)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# lm = LogisticRegression() 
# data = pd.read_csv("five_minute.csv", parse_dates = ["time"], index_col="time")
# data["returns"] = np.log(data.div(data.shift(1)))
# data.dropna(inplace = True)
# data["direction"] = np.sign(data.returns)
# lags = 5
# cols = []
# for lag in range(1, lags + 1):
#     col = "lag{}".format(lag)
#     data[col] = data.returns.shift(lag)
#     cols.append(col)
# data.dropna(inplace = True)
# print(data)  

# lm = LogisticRegression(C=1e6, max_iter = 1000000)
# lm.fit(data[cols], data.direction)
# data["pred"] = lm.predict(data[cols])
# data.pred.value_counts()
# hits = np.sign(data.direction * data.pred).value_counts() # type: ignore
# hit_ratio = hits[1.0] / sum(hits)
# print(hit_ratio)
# print(data)
# print(accuracy_score(y_true = data.direction, y_pred = data.pred) )


# Load and preprocess data
# data = pd.read_csv("test_set.csv", parse_dates=["time"], index_col="time")
# data["returns"] = np.log(data["price"] / data["price"].shift(1))  # Assuming 'price' column exists
# data["direction"] = np.sign(data["returns"])

# # Create lagged features
# lags = 5
# cols = []
# for lag in range(1, lags + 1):
#     col = f"lag{lag}"
#     data[col] = data["returns"].shift(lag)
#     cols.append(col)

# # Drop missing values
# data.dropna(inplace=True)

# # Normalize lagged features
# means = data[cols].mean()
# stand_devs = data[cols].std()
# data[cols] = (data[cols] - means) / stand_devs

# # Train logistic regression model
# lm = LogisticRegression()
# lm.fit(data[cols], data["direction"])

# # Predict market direction
# data["pred"] = lm.predict(data[cols])

# # Evaluate prediction accuracy
# hits = np.sign(data["direction"] * data["pred"]).value_counts() # pyright: ignore[reportAttributeAccessIssue]
# hit_ratio = hits.get(1.0, 0) / hits.sum()
# print("Hit Ratio:", hit_ratio)

# # Simulate strategy
# data["strategy"] = data["pred"] * data["returns"]
# data["creturns"] = data["returns"].cumsum().apply(np.exp)
# data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

# # Plot performance
# data[["creturns", "cstrategy"]].plot(figsize=(12, 8), title="Market vs Strategy Performance")
# plt.show()

# # Count trades
# data["trades"] = data["pred"].diff().fillna(0).abs()
# print("Trade Counts:\n", data["trades"].value_counts())

# # Final data snapshot
# print(data.tail()) 

# class MLBacktester():
#     ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
#     '''

#     def __init__(self, symbol, start, end, tc):
#         '''
#         Parameters
#         ----------
#         symbol: str
#             ticker symbol (instrument) to be backtested
#         start: str
#             start date for data import
#         end: str
#             end date for data import
#         tc: float
#             proportional transaction/trading costs per trade
#         '''
#         self.symbol = symbol
#         self.start = start
#         self.end = end
#         self.tc = tc
#         self.model = OneVsRestClassifier(LogisticRegression(C = 1e6, max_iter = 100000)) # new (from sklearn v. 1.7)
#         self.results = None
#         self.get_data()
    
#     def __repr__(self):
#         rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {})"
#         return rep.format(self.symbol, self.start, self.end, self.tc)
                             
#     def get_data(self):
#         ''' Imports the data from Yfinance.
#         '''
#         raw = yf.download(self.symbol, self.start, self.end)
#         raw = raw.dropna().copy() # pyright: ignore[reportOptionalMemberAccess]
#         raw = raw.rename(columns={"Close": "price"})
#         raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
#         self.data = raw
#         return raw
                             
#     def split_data(self, start, end):
#         ''' Splits the data into training set & test set.
#         '''
#         data = self.data.loc[start:end].copy()
#         return data
    
#     def prepare_features(self, start, end):
#         ''' Prepares the feature columns for training set and test set.
#         '''
#         self.data_subset = self.split_data(start, end)
#         self.feature_columns = []
#         for lag in range(1, self.lags + 1):
#             col = "lag{}".format(lag)
#             self.data_subset.loc[:, col] = self.data_subset["returns"].shift(lag)
#             self.feature_columns.append(col)
#         self.data_subset.dropna(inplace=True)

#     def scale_features(self, recalc = True): # Newly added
#         ''' Scales/Standardizes Features
#         '''
#         if recalc:
#             self.means = self.data_subset[self.feature_columns].mean()
#             self.stand_devs = self.data_subset[self.feature_columns].std()
        
#         self.data_subset[self.feature_columns] = (self.data_subset[self.feature_columns] - self.means) / self.stand_devs
        
#     def fit_model(self, start, end, lags):
#         ''' Fitting the ML Model.
#         '''
#         self.lags = lags 
#         self.prepare_features(start, end)
#         self.scale_features(recalc = True) # calculate mean & std of train set and scale train set
#         self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset["returns"]))
        
#     def test_strategy(self, train_ratio = 0.7, lags = 5):
#         ''' 
#         Backtests the ML-based strategy.
        
#         Parameters
#         ----------
#         train_ratio: float (between 0 and 1.0 excl.)
#             Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
#         lags: int
#             number of lags serving as model features.
#         '''
#         self.lags = lags
                  
#         # determining datetime for start, end and split (for training an testing period)
#         full_data = self.data.copy()
#         split_index = int(len(full_data) * train_ratio)
#         split_date = full_data.index[split_index-1]
#         train_start = full_data.index[0]
#         test_end = full_data.index[-1]
        
#         # fit the model on the training set
#         self.fit_model(train_start, split_date, lags)
        
#         # prepare the test set
#         self.prepare_features(split_date, test_end)
#         self.scale_features(recalc = False) # Newly added -> scale test set features with train set mean & std
                  
#         # make predictions on the test set
#         predict = self.model.predict(self.data_subset[self.feature_columns])
#         self.data_subset["pred"] = predict
        
#         # calculate Strategy Returns
#         self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
#         # determine the number of trades in each bar
#         self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
#         # subtract transaction/trading costs from pre-cost return
#         self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
#         # calculate cumulative returns for strategy & buy and hold
#         self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
#         self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
#         self.results = self.data_subset
        
#         perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
#         outperf = perf - self.results["creturns"].iloc[-1] # out-/underperformance of strategy
        
#         return round(perf, 6), round(outperf, 6)
        
#     def plot_results(self):
#         ''' Plots the performance of the trading strategy and compares to "buy and hold".
#         '''
#         if self.results is None:
#             print("Run test_strategy() first.")
#         else:
#             title = "Logistic Regression: {} | TC = {}".format(self.symbol, self.tc)
#             self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8)) 

# # Initialize the backtester
# ticker = MLBacktester("AAPL", "2022-01-01", "2023-01-01", 0.002)
# print(ticker)
# ticker.get_data()
# print(ticker.test_strategy(0.67, 7))
# ticker.plot_results() 
# plt.show()
# print(ticker.results)