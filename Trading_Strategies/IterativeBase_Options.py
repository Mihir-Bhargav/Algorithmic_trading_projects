
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_async import *  # type: ignore
plt.style.use("seaborn-v0_8")
from datetime import datetime


class IterativeBase():
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''
    def __init__(self, symbol, expiry, strike, right, amount, use_spread=True):
        self.symbol = symbol
        self.expiry = expiry
        self.strike = strike
        self.right = right
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.use_spread = use_spread

        self.ib = IB()

        if not self.ib.isConnected():
            try:
                self.ib.connect()
                print("Connected:", self.ib.isConnected())
            except Exception as e:
                print("IBKR connection failed:", e)

        self.get_data()

    def get_data(self):
        ib = self.ib

        contract = Option(
            symbol=self.symbol,
            lastTradeDateOrContractMonth=self.expiry,
            strike=self.strike,
            right=self.right,
            exchange='NSE',
            currency='INR',
            tradingClass='BANKNIFTY'
        )

        # REQUIRED
        ib.qualifyContracts(contract)

        data = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 M',
            barSizeSetting='5 mins',
            whatToShow='MIDPOINT',   # MIDPOINT often fails for options
            useRTH=True,
            formatDate=1
        )

        df = util.df(data)
        if df is None or df.empty:
            print("No data returned.")
            return None

        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

        df.rename(columns={"close": "price", "high": "High", "low": "Low"}, inplace=True)
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        df["spread"] = (df["High"] - df["Low"]) / 2
        df = df[["price", "spread", "returns"]].dropna()

        self.data = df
        return df


    def plot_data(self, cols = None):  
        ''' Plots the closing price for the symbol.
            '''
        df = self.data.copy()
        plt.figure(figsize=(12, 6))
        plt.title(f"{self.symbol}, Expiry: {self.expiry}, Strike: {self.strike}, {self.right}")
        plt.xlabel("Bar number")
        plt.ylabel("Price of option")
        plt.xticks(rotation=45)
        plt.plot(df)
        plt.tight_layout()
        plt.show()

    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date()) # type: ignore
        price = round(self.data.price.iloc[bar], 5)
        spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a buy order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread/2 # ask price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units # type: ignore
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a sell order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread/2 # bid price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -= units # type: ignore
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
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
        ''' Closes out a long or short position (go neutral).
        '''
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-") 




ticker = IterativeBase(
    symbol="BANKNIFTY",
    expiry="20251230",
    strike=59500,
    right="C",
    amount=100000
)

# Data is already fetched in __init__
df = ticker.get_data()
print(df)
df.to_csv("banknifty_20251230_59500_5m.csv", index=True)
print("CSV file saved successfully!")


# Note the frequency can be changes as wished. 