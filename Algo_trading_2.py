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
import FinancialInstrument as FI
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn-v0_8")

# # Sample data
# data = {
#     "Date": ["2025-09-01", "2025-09-02", "3 may 2025"],
#     "Price": ["150.5", "152.3", "149.8"]
# }

# df = pd.DataFrame(data)

# # Convert 'Date' to datetime
# df["Date"] = pd.to_datetime(df["Date"])
# print(df["Date"])

# # Convert 'Price' to float
# df["Price"] = pd.to_numeric(df["Price"])

# temp = pd.read_csv("temp.csv")
# print(temp.head(10))
# print(pd.resample("D"))

# stocks = pd.read_csv("aapl_fb.csv", parse_dates= ["Date"], index_col = "Date") 
# print(stocks)
# print(stocks.info())
# # 1005 enteries means that there are 1005 timestamps
# print(stocks.max())
# # The highest stock price for AAPL was 232 and for fb it was 217
# print(stocks.loc["2016-05-13"])
# # The prices were 90 and 119 for AAPL and FB respectivley. 
# september_stocks= stocks.loc["2017-09"]
# print(september_stocks.mean())
# print(stocks.resample("ME").last())

# print(stocks.mean())
# stocks_m  = stocks.resample("ME").mean() 
# print(stocks_m.head())
# The stock price was 82$. 

# start = "2014-10-01"
# end = "2021-05-31"
# symbol = ["BA", "MSFT", "TSLA", "ORCL", "AAPL"]
# df = yf.download(symbol, start, end)
# print(df)
# df.to_csv("multi_assets.csv") # type: ignore
# # Very powerful; creates a csv file. 
# ma = pd.read_csv("multi_assets.csv", header = [0,1], index_col = 0, parse_dates=[0])
# # print(ma)
# # print(ma.loc["2015"]) # type: ignore
# close = ma.Close.copy()
# close.BA.dropna().plot(figsize = (15,8))
# # plt.show()
# # print(close.iloc[0,0])
# relative_ratios_BA = close.BA.div(close.iloc[0,0]).mul(100) # type: ignore
# # print(relative_ratios_BA)
# # print(close.iloc[0])
# norm  = close.div(close.iloc[0].mul(1))
# # print(norm)
# This is a normalised price comparasion and the change factor is printed, given that the original investment = 1x. Tesla for example got 13 which means that stock 13x in the given time frame. 
# For Indian stocks add the suffix .NS

# msft = close.MSFT.dropna().to_frame().copy()
# msft.rename(columns = {"MSFT":"Price"}, inplace = True)
# msft["Returns"] = msft.Price.pct_change(periods = 1) 


# msft.Price.div(msft.P_lag1) - 1 # type: ignore # Alternative 1
# msft["P_diff2"] = msft.Price.diff(periods = 1)  # Alternative 2
# print(msft.drop(columns = ["P_lag1", "P_diff", "P_diff2"], inplace = True))

# start = "2014-10-01"
# end = "2021-05-31"
# symbol = "MSFT"
# msft_csv = yf.download(symbol, start, end)

# msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
# print(msft)
# msft.plot(figsize=(15,8), fontsize=13)
# plt.legend(fontsize=13)
# plt.show()
# mu = msft.Returns.mean() 
# # arithmetic mean return -> Reward
# sigma = msft.Returns.std() # standard deviation of returns -> Risk/Volatility
# print(mu, sigma)
# print(np.sqrt(msft["Returns"].var())) # type: ignore

# start = "2014-10-01"
# end = "2021-05-31"
# symbol = "BTC-USD"
# BTC_csv = yf.download(symbol, start, end)

# BTC_csv.to_csv("btc.csv") # type: ignore
# btc = pd.read_csv("btc.csv")
# btc["Open"] = pd.to_numeric(btc["Open"], errors="coerce" )
# btc["Close"] = pd.to_numeric(btc["Close"], errors="coerce")
# # print(btc)
# btc["Returns"] = (btc["Open"] - btc["Close"]) / btc["Open"]
# btc["Price"] = btc["Close"]
# BTC_price = btc["Price"]
# # print(btc["Returns"])
# print(btc["Returns"].mean())
# BTC_price.plot(figsize=(15,8), fontsize=13)
# plt.legend(fontsize=13)
# plt.xlabel = "Time"
# plt.ylabel = "BTC-Price"
# plt.title = "Bitcoin price chart"
# # plt.show() 
# # print(btc.Close[0])
# clean_close = btc["Close"].dropna()
# multiple = clean_close.iloc[-1] / clean_close.iloc[0]
# print(multiple)
# percent_increase = ((multiple- 1)*100).round()
# print(f"{percent_increase} is the % change.")
# # Good work, you created your own returns column using simple math!

# start = btc.index[0]
# print(start)
# end = btc.index[-1]
# print(end)
# td = end - start
# td_years = td/365.25
# print(td_years)
# cagr = multiple**(1/td_years)
# print(cagr)
# n = len(btc)
# geomean = (multiple**(1/n) - 1)
# print(geomean)

# MACSv = pd.read_csv("multi_assets.csv", header=[0, 1], index_col=0, parse_dates=True)
# print(MACSv)
# BA_close = MACSv["Close"]["BA"].dropna().to_frame().copy()
# print(BA_close)
# multiple = ((BA_close.iloc[-1] - BA_close.iloc[0])/ BA_close.iloc[0])
# print(multiple)
# n=len(BA_close)
# geomean = (multiple**(1/n)-1)
# print(f"The geometric mean is {geomean}")

# start = BA_close.index[0]
# end = BA_close.index[-1]
# td = end - start 
# td_years = td.days / 365.25  
# print(td_years)
# cagr = (multiple**(1/td_years))
# print(f"The compound rate of growth is {cagr}. ")

# PV = 100
# r = 0.08
# n = 1 
# m = 4
# print(PV + (r + n))
# FV = PV * (1 + r)**n
# effective_annual_rate = (FV / PV)**(1/n) - 1 
# print(effective_annual_rate)

# # For the effective ir when m!=1
# FV = PV * (1 + r/m)**(n*m)
# print(FV)
# effective_annual_rate = (FV / PV)**(1/n) - 1 
# print(effective_annual_rate)

# FV = PV * np.exp(n * r) # exact math with e (euler number)
# print(FV)
# euler = np.exp(1)
# print(euler)
# print(PV * euler**(n * r))
# effective_annual_rate = np.exp(r) - 1 # Alt 2
# print(effective_annual_rate)
# r = np.log(FV / PV) # inverse calculation -> use log (Alt 1)
# print(r)

# msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
# msft["log_ret"] = np.log(msft.Price / msft.Price.shift()) # daily log returns
# print(msft)
# print(msft.describe())

# df = pd.DataFrame(data = [100, 50, 90], columns = ["Price"])
# df["SR"] = df.Price.pct_change() # simple returns
# df["LR"] = np.log(df.Price / df.Price.shift()) # log returns 
# print(df["LR"], df["SR"])
# periods = df.SR.count()
# mean_sr = df.SR.mean()
# print(100 * (1 + mean_sr)**periods) # wrong!!!
# It prints 132, when the actual value was 90.
# geo_mean = (1 + df.SR).prod()**(1 / periods) - 1
# print(100 * (1 + geo_mean)**periods) 
# prints 89.99, correct, but there is a faster way using e. 

# sum_lr = df.LR.sum()
# print(100 * np.exp(sum_lr))

# msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"])
# msft["log_ret"] = np.log(msft.Price / msft.Price.shift())
# print(msft.Returns.add(1).prod()) # compounding simple returns ("compound returns")
# print(np.exp(msft.log_ret.sum()))  # adding log returns ("cumulative returns")

# They both return the same thing... 

# print(msft.Returns.add(1).cumprod()) # compounding simple returns ("compound returns")
# print(np.exp(msft.log_ret.cumsum())) # adding log returns ("cumulative returns")

# print((msft.Price.iloc[-1]/msft.Price.iloc[0])**(1/((msft.index[-1] - msft.index[0]).days / 365.25)) - 1) # use iloc

# trading_days_year = msft.Returns.count() / ((msft.index[-1] - msft.index[0]).days / 365.25)

# print(np.exp(msft.log_ret.mean() * trading_days_year) - 1 )# correct with mean of daily log returns!


# close = pd.read_csv("multi_assets.csv", header=[0, 1, 2], index_col=3, parse_dates=True)
# close.info()
# returns = close.apply(lambda x: np.log(x / x.shift()))
# print(returns)
# print(returns.info())
# print(returns.describe())
# <<<<<Threw a MILLION ERRORS, COULDN't FIGURE WHY>>>>

# Load data
# data = pd.read_csv('multi_assets.csv')

# # Set the correct column names based on the header row
# data.columns = ['Date', 'AAPL_Close', 'BA_Close', 'MSFT_Close', 'ORCL_Close', 'TSLA_Close']

# # Convert relevant columns to numeric
# for col in data.columns[1:]:  # Skip the 'Date' column
#     data[col] = pd.to_numeric(data[col], errors='coerce')

# # Drop rows with NaN values
# data = data.dropna()

# # Check if the data looks correct
# print(data.head())

# # Calculate log returns for a specific instrument, for example, AAPL
# data['AAPL_Log_Returns'] = np.log(data['AAPL_Close'] / data['AAPL_Close'].shift(1))

# # Drop NaN resulting from log return calculation
# data = data.dropna()

# # Calculate mean and standard deviation of log returns
# mean_returns = data['AAPL_Log_Returns'].mean()
# std_dev = data['AAPL_Log_Returns'].std()

# # Create a summary DataFrame
# summary = pd.DataFrame({'Mean Return': [mean_returns], 'Standard Deviation': [std_dev]})

# # Mean-Variance Scatter Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(std_dev, mean_returns)
# plt.xlabel('Standard Deviation (Risk)')
# plt.ylabel('Mean Return')
# plt.title('Risk vs Return Analysis for AAPL')
# plt.grid()
# plt.show()

# print(summary)

# close = pd.read_csv("close.csv", index_col = "Date", parse_dates = ["Date"])
# close.info()
# np.log(close / close.shift()).info() # keep NaN
# close.apply(lambda x: np.log(x.dropna() / x.dropna().shift())).info() # remove NaN
# returns = close.apply(lambda x: np.log(x.dropna() / x.dropna().shift()))
# print(returns)
# returns.info()
# returns.describe()
# summary = returns.agg(["mean", "std"]).T
# print(summary)
# summary.columns = ["Mean", "Std"]
# print(summary)
# summary.plot(kind = "scatter", x = "Std", y = "Mean", figsize = (15,12), s = 50, fontsize = 15)
# for i in summary.index:
#     plt.annotate(i, xy=(summary.loc[i, "Std"]+0.00005, summary.loc[i, "Mean"]+0.00005), size = 15)
# plt.xlabel("Risk (std)", fontsize = 15)
# plt.ylabel("Mean Return", fontsize = 15)
# plt.title("Mean-Variance Analysis", fontsize = 20)
# plt.show()

# Pathetic dealing with errors. You will have to learn how to deal with even complicated errors like this. Moreover, find a stable defense system if a 100% autonomy is wished.

# msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"], usecols = ["Date", "Price"])
# msft["log_ret"] = np.log(msft.Price / msft.Price.shift()) 
# # msft.log_ret.plot(kind = "hist", figsize = (15 ,8), bins = 100, fontsize = 15, density = False) # Frequency Distribution of log returns
# # plt.xlabel("Daily Returns", fontsize = 15)
# # plt.ylabel("Frequency", fontsize = 15)
# # plt.title("Frequency Distribution of Returns", fontsize = 20)
# # plt.show()
# mu = msft.log_ret.mean()
# print(mu)
# sigma = msft.log_ret.std()
# print(sigma)
# import scipy.stats as stats
# # print(stats.skew(msft.log_ret.dropna()))

# x = np.linspace(msft.log_ret.min(), msft.log_ret.max(), 10000)
# y = stats.norm.pdf(x, loc = mu, scale = sigma) # creating y values a for normal distribution with mu, sigma
# plt.figure(figsize = (20, 8))
# plt.hist(msft.log_ret, bins = 500, density = True, label = "Frequency Distribution of daily Returns (MSFT)")
# plt.plot(x, y, linewidth = 3, color = "red", label = "Normal Distribution")
# plt.title("Normal Distribution", fontsize = 20)
# plt.xlabel("Daily Returns", fontsize = 15)
# plt.ylabel("pdf", fontsize = 15)
# plt.legend(fontsize = 15)
# # plt.show()


# msft = pd.read_csv("msft.csv", index_col = "Date", parse_dates = ["Date"], usecols = ["Date", "Price"])
# msft["log_ret"] = np.log(msft.Price / msft.Price.shift())
# print(msft.log_ret.agg(["mean", "std"]))
# ann_mu = msft.log_ret.mean() * 252 
# print(ann_mu)
# cagr = np.exp(ann_mu) - 1 # don´t mix up with cagr
# print(cagr)
# ann_std = msft.log_ret.std() * np.sqrt(252) 
# print(ann_std)

# window = 252
# msft.log_ret.rolling(window = 252)
# msft.log_ret.rolling(window = 252).sum()
# roll_mean = msft.log_ret.rolling(window = 252).mean() * 252 # Alt 2 
# print(roll_mean)
# roll_mean.iloc[250:]
# roll_std = msft.log_ret.rolling(window = 252).std() * np.sqrt(252)
# roll_std.plot(figsize = (12, 8))
# # plt.show()
# roll_mean.plot(figsize = (12, 8))
# roll_std.plot()
# # plt.show()

# sma_window = 50
# msft.Price.plot(figsize = (12, 8))
# msft.Price.rolling(sma_window).mean().plot()
# # plt.show()
# # -> For simple returns: long position returns != short position returns * (-1)
# # -> For log returns: long position returns == short position returns * (-1)

# prices = pd.DataFrame(data = {"Asset_A": [100, 112], "Asset_B":[100, 104]}, index = [0, 1])
# prices["Total"] = prices.Asset_A + prices.Asset_B
# print(prices)
# returns = prices.pct_change() # simple returns
# print(returns)
# log_returns = np.log(prices / prices.shift()) # log returns
# print(log_returns)

# Very important!! Levered trades amplify the stakes, which means higher reqrads but also more potential downside. The volatility, is also higher. There are breakevens, so look into that when trading with margin. Take Home:
# 1. With Leverage you can (theoretically)lose more than the initial Margin(in practice: margin call / margin closeout before)
# 2. Even for (highly) profitable instruments: "The more leverage the better" does not hold__.
# 3. It´s a two edged (non-symmetrical) sword: Leverage amplifies losses more than it amplifies gains.

# SP = pd.read_excel("SP500.xls", parse_dates = ["Date"], index_col="Date", usecols = "A, C:E")
# # You can also use the sheet_name to chose a specific column of Data. 
# print(SP)
# SP.to_csv("SP500.csv")

# Merging Times Series and dataframes

# stocks = pd.read_csv("stocks.csv", header=[0,1], index_col=[0], parse_dates=[0]).Close
# print(stocks)
# print(stocks.columns.get_level_values(0).unique())
# # It's literally that easy to filter out columns. Struggled a hell lot last time with errors for some reason. 
# # print(stocks)
# aapl = stocks.loc["2010-01-01" : "2014-12-31", "AAPL"].to_frame()
# ba = stocks.loc["2012-01-01" : "2016-12-31", "BA"].to_frame()
# aapl["BA"] = ba.BA 
# print(aapl)
# print(aapl.dropna())
# print(ba.reindex(aapl.index).dropna())

# print(close.index.month)
# print(close.index.year)

# stocks = pd.read_csv("stocks.csv", header=[0,1], index_col=0, parse_dates=[0])
# close = stocks.xs("Close", axis=1, level=0).copy()
# print(stocks.head()) 
# print(close.head())
# print(close.info())
# print(close.index.day)
# print(close.index.day_name())
# all_days = pd.date_range(start = "2009-12-31", end = "2019-02-06", freq = "D") 
# print(all_days)
# close = close.reindex(all_days)
# print(close.head(20))
# # As we saw, there were a lot of empty values. There are 2 alternatives. Frontfill fills the gaps infront with the latest defined value while the backfill value fills the gaps with the nearest after the gap defined data. 
# close.ffill(inplace=True)
# print(close)
# print(close.interpolate(inplace=True))

# ge = pd.read_csv("GE_prices.csv", parse_dates= ["date"], index_col= "date")
# # print(ge.head(30))
# # print(ge.index.tz)
# # print(ge.tz_localize("UTC"))
# # print(ge.tz_localize("America/New_York"))
# ge = ge.tz_localize("America/New_York") 
# # print(ge.tz_convert("America/Los_Angeles"))
# ge_la = ge.tz_convert("America/Los_Angeles") 
# # print(ge_la.head())
# comb = pd.concat([ge, ge_la], axis = 1)
# print(comb.head())
# print(pytz.common_timezones)

# class FinancialInstrument():
#     ''' Class for analyzing Financial Instruments like stocks.

#     Attributes
#     ==========
#     ticker: str
#         ticker symbol with which to work with
#     start: str
#         start date for data retrieval
#     end: str
#         end date for data retrieval

#     Methods
#     =======
#     get_data:
#         retrieves daily price data (from yahoo finance) and prepares the data
#     log_returns:
#         calculates log returns
#     plot_prices:
#         creates a price chart
#     plot_returns:
#         plots log returns either as time series ("ts") or histogram ("hist")
#     set_ticker:
#         sets a new ticker
#     mean_return:
#         calculates mean return
#     std_returns:
#         calculates the standard deviation of returns (risk)
#     annualized_perf:
#         calculates annulized return and risk
#     '''
    
#     def __init__(self, ticker, start, end):
#         self._ticker = ticker
#         self.start = start
#         self.end = end
#         self.get_data()
#         self.log_returns()
    
#     def __repr__(self): 
#         return "FinancialInstrument(ticker = {}, start = {}, end = {})".format(self.ticker, 
#                                                                                self.start, self.end)
#     def get_data(self):
#         ''' retrieves (from yahoo finance) and prepares the data
#         '''
#         raw = yf.download(self.ticker, self.start, self.end, multi_level_index = False).Close.to_frame() # type: ignore
#         raw.rename(columns = {"Close":"price"}, inplace = True)
#         self.data = raw
        
#     def log_returns(self):
#         '''calculates log returns
#         '''
#         self.data["log_returns"] = np.log(self.data.price/self.data.price.shift(1))
        
#     def plot_prices(self):
#         ''' creates a price chart
#         '''
#         self.data.price.plot(figsize = (12, 8))
#         plt.title("Price Chart: {}".format(self.ticker), fontsize = 15)
    
#     def plot_returns(self, kind = "ts"):
#         ''' plots log returns either as time series ("ts") or histogram ("hist")
#         '''
#         if kind == "ts":
#             self.data.log_returns.plot(figsize = (12, 8))
#             plt.title("Returns: {}".format(self.ticker), fontsize = 15)
#         elif kind == "hist":
#             self.data.log_returns.hist(figsize = (12, 8), bins = int(np.sqrt(len(self.data))))
#             plt.title("Frequency of Returns: {}".format(self.ticker), fontsize = 15)
    
#     def set_ticker(self, ticker = None):
#         '''sets a new ticker
#         '''
#         if ticker is not None:
#             self.ticker = ticker
#             self.get_data()
#             self.log_returns()
            
#     def mean_return(self, freq = None):
#         '''calculates mean return
#         '''
#         if freq is None:
#             return self.data.log_returns.mean()
#         else:
#             resampled_price = self.data.price.resample(freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.mean()
    
#     def std_returns(self, freq = None):
#         '''calculates the standard deviation of returns (risk)
#         '''
#         if freq is None:
#             return self.data.log_returns.std()
#         else:
#             resampled_price = self.data.price.resample(freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.std()
        
#     def annualized_perf(self):
#         '''calculates annulized return and risk
#         '''
#         mean_return = round(self.data.log_returns.mean() * 252, 3)
#         risk = round(self.data.log_returns.std() * np.sqrt(252), 3)
#         print("Return: {} | Risk: {}".format(mean_return, risk)) 


# stock = FinancialInstrument(ticker = "^NSEBANK", start = "2015-01-01", end =  "2019-12-31" ) 

# print(stock)
# stock.plot_prices()
# plt.show()
# print(stock.mean_return(freq="W"))
# print(stock.std_returns(freq="W"))
# print(stock.annualized_perf())
# Extremely cool right, but this doesn't come out of nowhere. I just copy pasted the class finacncial instrument, but let's see how to make our own classes. 


# stock = FinancialInstrument(ticker = "^NSEBANK", start = "2015-01-01", end =  "2019-12-31" )

# raw = yf.download("AAPL", "2015-01-01", "2019-12-31", multi_level_index = False).Close.to_frame() # type: ignore
# raw.rename(columns = {"Close":"price"}, inplace = True)
# print(raw) 

# Now we could have it like this, but to have it more versatile, we could add a self tag. This reminds me of usign the {input} instead of a specific thing. 

# class FinancialInstrumentBase():
#     def __init__(self, ticker, start, end):
#         self._ticker = ticker
#         self.start = start
#         self.end = end
#         self.get_data()
#         self.log_returns()
#     def __repr__(self):
#         return "FinancialInstrument(ticker = {}, start = {}, end = {})".format(self._ticker, self.start, self.end)
#     def get_data(self):
#         raw = yf.download(self._ticker, self.start, self.end, multi_level_index = False).Close.to_frame() # type: ignore
#         raw.rename(columns = {"Close":"price"}, inplace = True)
#         self.data = raw
#     def log_returns(self): 
#         self.data["log_returns"] = np.log(self.data.price/self.data.price.shift(1))
#     def plot_prices(self):
#         stock.data.price.plot() 
#         plt.title("Price Chart: {}".format(self._ticker), fontsize = 15)
#     def plot_returns(self, kind = "ts"):
#         if kind == "ts":
#             self.data.log_returns.plot(figsize = (12, 8))
#             plt.title("Returns: {}".format(self._ticker), fontsize = 15)
#         elif kind == "hist":
#             self.data.log_returns.hist(figsize = (12, 8), bins = int(np.sqrt(len(self.data))))
#             plt.title("Frequency of Returns: {}".format(self._ticker), fontsize = 15)  
#     def set_ticker(self, ticker=None):
#         if ticker is not None: 
#             self._ticker = ticker
#             self.get_data()
#             self.log_returns() 
#     def mean_return(self, freq = None):
#         if freq is None:
#             return self.data.log_returns.mean()
#         else:
#             resampled_price = self.data.price.resample(freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.mean()
#     def std_returns(self, freq = None):
#         if freq is None:
#             return self.data.log_returns.std()
#         else:
#             resampled_price = self.data.price.resample(freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.std()
#     def annualized_perf(self):
#         mean_return = round(self.data.log_returns.mean() * 252, 3)
#         risk = round(self.data.log_returns.std() * np.sqrt(252), 3)
#         print("Return: {} | Risk: {}".format(mean_return, risk))

# stock = FinancialInstrument("AAPL", "2015-01-01", "2019-12-31")

# As demonstrated, self should be passed as an argument, nothing complex, but common error. 
# print(stock.data)
# print(stock) 
# stock.plot_returns(kind="ts")
# plt.show()
# We can plot (manually) the grapgh using 
# stock.data.price.plot()
# plt.show() 
# But we can also automate this by adding this to the class FinancialInstrument>>>
# We can also say: 
# stock.plot_prices() 
# plt.show()
# This overrides the "AAPl". To fix this or make it slightly harder we can use protected Attributes. 
# This can be done with stock._ticker. Now, this prints AAPl stock price. SMall system to avoid confusion. 
# print(stock) 
# stock.plot_prices() 
# plt.show()
# stock.set_ticker("GE")
# stock.plot_prices()
# plt.show() 
# Testing all aspects of the class.... 
# print(stock)

# # Print raw data
# print("\n📊 Price and Log Returns Data:")
# print(stock.data.head())

# # Print mean return (daily)
# print("\n📈 Mean Daily Log Return:")
# print(stock.mean_return())

# # Print mean return (monthly)
# print("\n📈 Mean Monthly Log Return:")
# print(stock.mean_return(freq="ME"))

# # Print standard deviation (daily)
# print("\n📉 Daily Volatility:")
# print(stock.std_returns())

# # Print standard deviation (monthly)
# print("\n📉 Monthly Volatility:")
# print(stock.std_returns(freq="ME"))

# # Print annualized performance
# print("\n📊 Annualized Performance:")
# stock.annualized_perf()

# # Plot price chart
# print("\n📈 Plotting Price Chart:")
# stock.plot_prices()
# plt.show()

# # Plot time series of returns
# print("\n📈 Plotting Returns Time Series:")
# stock.plot_returns(kind="ts")
# plt.show()

# # Plot histogram of returns
# print("\n📊 Plotting Returns Histogram:")
# stock.plot_returns(kind="hist")
# plt.show()
# # Everything works!

# When we have several classes, with similar or a few common methods, it would be very time consuming to rewrite or even copy and paste them every time. That is why we can instead the pass function and the new class "inherits" all the methods. 
# Here, RiskReturn is the child of the FinancialInstrument Class. 


# class RiskReturn(FinancialInstrumentBase): 
#     def __init__(self, ticker, start, end, freq = None):
#         self.freq = freq
#         super().__init__(ticker, start, end)
# # The super enables the code from the parent to be used.     
#     def __repr__(self): 
#         return "RiskReturn(ticker = {}, start = {}, end = {})".format(self._ticker, 
#                                                                           self.start, self.end)
#     def mean_return(self):
#         if self.freq is None:
#             return self.data.log_returns.mean()
#         else:
#             resampled_price = self.data.price.resample(self.freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.mean()
    
#     def std_returns(self):
#         if self.freq is None:
#             return self.data.log_returns.std()
#         else:
#             resampled_price = self.data.price.resample(self.freq).last()
#             resampled_returns = np.log(resampled_price / resampled_price.shift(1))
#             return resampled_returns.std()
        
#     def annualized_perf(self):
#         mean_return = round(self.data.log_returns.mean() * 252, 3)
#         risk = round(self.data.log_returns.std() * np.sqrt(252), 3)
#         print("Return: {} | Risk: {}".format(mean_return, risk))

# stock = RiskReturn("nvda", "2015-01-01", "202-12-31", freq = "W")

# stock.plot_prices()
# plt.show()

# stocks = FI.FinancialInstrument("NVDA", start="2010-01-01", end="2024-01-01")
# print(stocks.mean_return())
# print(stocks.data)
# stocks.plot_prices() 
# plt.show()

# class rectangleCalculation(): 
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#     def __repr__ (self):
#         return "rectangleCalculation(width ={}, height{})".format(self.a,self.b)
#     def cal_area(self): 
#         return self.a * self.b
#     def cal_hypot(self): 
#         return math.sqrt( self.a**2 + self.b**2)
#     def cal_perimeter(self): 
#         return 2 * (self.a + self.b)     


# rec = rectangleCalculation(a= 10, b = 15)
# print(rec.cal_area())
# print(rec.cal_hypot())
# print(rec.cal_perimeter())

# import random
# import math
# count = 0


# count = 0

# for i in range(0, 101):
#     random_dec = random.random()
#     one_to_six = math.floor(random_dec * 6 + 1)

#     random_dec_2 = random.random()
#     one_to_six_2 = math.floor(random_dec_2 * 6 + 1)

#     if one_to_six == one_to_six_2:
#         count += 1
#         print(f"Match #{count}: {one_to_six} == {one_to_six_2}") 


print("GITHUB is working!!!!. New changes made...")