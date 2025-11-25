
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_async import * 
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
        try:
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            print("Connected:", self.ib.isConnected())
        except Exception as e:
            print("IBKR connection failed:", e)

        self.get_data()

    def get_data(self):
        ib = self.ib
        
        symbol  = self.symbol
        expiry  = self.expiry
        strike  = self.strike
        right   = self.right

        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='NSE',
            currency='INR'
        )

        data = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 M',
            barSizeSetting='3 mins',
            whatToShow='MIDPOINT',
            useRTH=True
        )

        df = util.df(data)
        if df is None or df.empty:
            print("No data returned.")
            return

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
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date())
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
        self.units += units
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
        self.units -= units
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


ticker = IterativeBase("BANKNIFTY", "20251125", "59000", "C", 28000, True)
df = ticker.get_data()

print("=== DATA PREVIEW ===")
print(df.head())

print("\n=== TEST get_values ===")
for bar in range(3):  # test first 3 bars
    date, price, spread = ticker.get_values(bar)
    print(f"Bar {bar}: date={date}, price={price}, spread={spread}")

print("\n=== print_current_balance ===")
ticker.print_current_balance(0)

print("\n=== buy_instrument ===")
ticker.buy_instrument(bar=0, amount=5000)  # buy with a portion of initial balance
ticker.print_current_balance(0)
ticker.print_current_position_value(0)
ticker.print_current_nav(0)

print("\n=== sell_instrument ===")
ticker.sell_instrument(bar=1, units=10)  # sell some units
ticker.print_current_balance(1)
ticker.print_current_position_value(1)
ticker.print_current_nav(1)

print("\n=== buying and selling more ===")
ticker.buy_instrument(bar=2, units=50)
ticker.sell_instrument(bar=2, amount=2000)
ticker.print_current_balance(2)
ticker.print_current_position_value(2)
ticker.print_current_nav(2)

print("\n=== close_pos ===")
ticker.close_pos(bar=204)  # close remaining units
ticker.print_current_balance(2)
ticker.print_current_position_value(2)
ticker.print_current_nav(2)

print("\n===plot_data ===")
ticker.plot_data() 


