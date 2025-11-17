
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use("seaborn-v0_8")

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
        # Depending on what timeframe, chose between get_data and get_data_intraday. 

    def get_data(self):
        raw = yf.download(self.symbol, self.start, self.end)
        raw = raw.copy().dropna() # type: ignore
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={"Close": "price"}, inplace=True)
        raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
        raw["spread"] = 0.005 * (raw["High"] - raw["Low"])
        # raw = raw[["price", "spread", "returns"]].dropna().copy()
        self.data = raw
        return raw

    def plot_data(self, cols=None):
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)
        plt.show()

    def get_values(self, bar):
        # date = str(self.data.index[bar].date()) # pyright: ignore[reportAttributeAccessIssue]
        # price = float(self.data.price.iloc[bar])  # convert to scalar float
        # spread = float(self.data.spread.iloc[bar])  # convert to scalar float
        # return date, price, spread
        date = f"Row {bar}"  # placeholder string instead of datetime
        price = float(self.data['[C_LAST]'].iloc[bar])
        spread = 0.002  # options: set to 0 or a small fixed cost
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
