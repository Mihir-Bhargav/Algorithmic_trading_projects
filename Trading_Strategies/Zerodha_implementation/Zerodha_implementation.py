import numpy as np 
import scipy as sp 
import math 
import pandas as pd 
import sympy as smp 
from IterativeBase import *
import logging
from kiteconnect import KiteConnect 
logging.basicConfig(level=logging.DEBUG)


def connect_Zerodha():
    kite = KiteConnect(api_key="p6vzf3je78hvp6l1")
    print(kite.login_url())
    data = kite.generate_session("JXgIFE3KpDkppj5jLoAISUHZF19XmCLl", api_secret="7ddoafhnna3xvstatr1e50w8bqgs0twb")
    kite.set_access_token(data["access_token"])
    profile = kite.profile()
    print(profile)

class IterativeBacktest(IterativeBase):
    
    def go_long(self, bar, units=None, amount=None): 
        if self.position == -1:
            self.buy_instrument(bar, units=-self.units)  # close short
            self.position = 0
        if units: 
            self.buy_instrument(bar, units=units)
            self.position = 1
        elif amount: 
            if amount == "all": 
                amount = self.current_balance
            self.buy_instrument(bar, amount=amount)
            self.position = 1
    
    def go_short(self, bar, units=None, amount=None): 
        if self.position == 1: 
            self.sell_instrument(bar, units=self.units)  # close long
            self.position = 0
        if units: 
            self.sell_instrument(bar, units=units)
            self.position = -1
        elif amount:
            if amount == "all": 
                amount = self.current_balance
            self.sell_instrument(bar, amount=amount)
            self.position = -1

    def test_Derivative_strategy(self, tc=0.001):
        """Iterative backtest of the smoothed 2nd derivative strategy."""
        df = self.data.copy()
        t = 1 / 252  # Avoid timedelta dtype issues

        # --- Compute smoothed 2nd derivative ---
        df["price_smooth"] = df["price"].rolling(window=5).mean()
        x = np.arange(len(df)) * t
        f = df["price_smooth"].values
        dfdx = np.gradient(f, x) # pyright: ignore[reportArgumentType]
        d2fdx2 = np.gradient(dfdx, x)
        df["second_der_smooth"] = 300 * d2fdx2
        df["signal"] = np.where(df["second_der_smooth"] > 0, 1, -1)

        print("\nStarting derivative-based iterative backtest...")
        print(75 * "-")

        self.trades = 0
        self.position = 0
        self.units = 0
        trades_today = 0

        for bar in range(1, len(df)):
            date, price, spread = self.get_values(bar)
            sig = df["signal"].iloc[bar]

            if sig == 1 and self.position <= 0:  # Go long
                self.go_long(bar, amount="all")
                trades_today += 1
                self.trades += 1
                print(f"{date} | LONG signal | Price: {price:.2f} | Trades today: {trades_today}")
                self.print_current_nav(bar)

            elif sig == -1 and self.position >= 0:  # Go short
                self.go_short(bar, amount="all")
                trades_today += 1
                self.trades += 1
                print(f"{date} | SHORT signal | Price: {price:.2f} | Trades today: {trades_today}")
                self.print_current_nav(bar)

        # --- Close final position ---
        self.close_pos(len(df) - 1)

        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        print(f"\nFinal Performance: {perf:.2f}% | Total Trades: {self.trades}")
        print(75 * "-")

        return df
    
ticker = IterativeBacktest("AAPL", "2025-01-01", "2025-10-25", 10000, use_spread=True)
ticker.test_Derivative_strategy()


# class IterativeBacktest(IterativeBase):
    
#     def go_long(self, bar, units=None, amount=None): 
#         if self.position == -1:
#             self.buy_instrument(bar, units=-self.units)  # close short
#             self.position = 0
#         if units: 
#             self.buy_instrument(bar, units=units)
#             self.position = 1
#         elif amount: 
#             if amount == "all": 
#                 amount = self.current_balance
#             self.buy_instrument(bar, amount=amount)
#             self.position = 1
    
#     def go_short(self, bar, units=None, amount=None): 
#         if self.position == 1: 
#             self.sell_instrument(bar, units=self.units)  # close long
#             self.position = 0
#         if units: 
#             self.sell_instrument(bar, units=units)
#             self.position = -1
#         elif amount:
#             if amount == "all": 
#                 amount = self.current_balance
#             self.sell_instrument(bar, amount=amount)
#             self.position = -1

#     def test_Derivative_strategy(self, tc=0.001):
#         """Iterative backtest of the smoothed 2nd derivative strategy with SMA/RSI filter on 5-min bars."""
#         df = self.data.copy()
#         t = 1 / 252  # avoid timedelta issues

#         # --- Compute smoothed 2nd derivative ---
#         df["price_smooth"] = df["price"].rolling(window=5).mean()
#         x = np.arange(len(df)) * t
#         f = df["price_smooth"].values
#         dfdx = np.gradient(f, x) # pyright: ignore[reportArgumentType]
#         d2fdx2 = np.gradient(dfdx, x)
#         df["second_der_smooth"] = 300 * d2fdx2
#         df["signal"] = np.where(df["second_der_smooth"] > 0, 1, -1)

#         # --- Compute SMA and RSI ---
#         SMA_PERIOD = 10
#         RSI_PERIOD = 14
#         df["SMA"] = df["price"].rolling(SMA_PERIOD).mean()

#         delta = df["price"].diff()
#         up = delta.clip(lower=0)
#         down = -delta.clip(upper=0)
#         roll_up = up.rolling(RSI_PERIOD).mean()
#         roll_down = down.rolling(RSI_PERIOD).mean()
#         RS = roll_up / roll_down
#         df["RSI"] = 100 - (100 / (1 + RS))

#         print("\nStarting derivative-based iterative backtest with SMA/RSI filter...")
#         print(75 * "-")

#         self.trades = 0
#         self.position = 0
#         self.units = 0
#         trades_today = 0
#         MAX_TRADES_PER_DAY = 10
#         current_day = None

#         for bar in range(1, len(df)):
#             date, price, spread = self.get_values(bar)
#             # ensure date is datetime
#             if isinstance(date, str):
#                 date = pd.to_datetime(date)
#             day = date.date()
#             sig = df["signal"].iloc[bar]

#             # reset daily counter at start of new day
#             if day != current_day:
#                 current_day = day
#                 trades_today = 0

#             if trades_today >= MAX_TRADES_PER_DAY:
#                 continue

#             # --- SMA/RSI soft filter ---
#             sma_filter = (price > 0.995 * df["SMA"].iloc[bar]) if sig == 1 else (price < 1.005 * df["SMA"].iloc[bar])
#             rsi_filter = (df["RSI"].iloc[bar] > 45) if sig == 1 else (df["RSI"].iloc[bar] < 55)

#             if sig == 1 and self.position <= 0 and sma_filter and rsi_filter:  # Go long
#                 self.go_long(bar, amount="all")
#                 trades_today += 1
#                 self.trades += 1
#                 print(f"{date} | LONG signal | Price: {price:.2f} | Trades today: {trades_today}")
#                 self.print_current_nav(bar)

#             elif sig == -1 and self.position >= 0 and sma_filter and rsi_filter:  # Go short
#                 self.go_short(bar, amount="all")
#                 trades_today += 1
#                 self.trades += 1
#                 print(f"{date} | SHORT signal | Price: {price:.2f} | Trades today: {trades_today}")
#                 self.print_current_nav(bar)

#         # --- Close final position ---
#         self.close_pos(len(df) - 1)
#         perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
#         print(f"\nFinal Performance: {perf:.2f}% | Total Trades: {self.trades}")
#         print(75 * "-")
#         return df
    
# ticker = IterativeBacktest("GC=F", "2025-10-10", "2025-10-27", 10000, use_spread=True)
# ticker.test_Derivative_strategy()

