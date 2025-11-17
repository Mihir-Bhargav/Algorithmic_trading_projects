import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import ta as ta
import seaborn as sns
from ib_async import *  # type: ignore
import time

class Black_Scholes_Backtest:
    def __init__(self, symbol, start, end):
        self.start = start
        self.end = end
        self.symbol = symbol
        self.data = None  # Initialize data attribute
        self.ib = IB()                 # instance
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        print("Connected:", self.ib.isConnected())

    def get_data(self):
        raw = yf.download(self.symbol, self.start, self.end, interval="2m")
        raw = raw[['Close']].copy().dropna() # pyright: ignore[reportOptionalSubscript]
        raw.rename(columns={'Close': 'price'}, inplace=True)
        self.data = raw
        return raw

    def black_scholes_call(self, S, K, sigma, r, t):
        d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
        d2 = d1 - (sigma * np.sqrt(t))
        C = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        return C

    def black_scholes_put(self, S, K, sigma, r, t):
        d1 = (np.log(S / K) + (r + ((sigma ** 2) / 2)) * t) / (sigma * np.sqrt(t))
        d2 = d1 - (sigma * np.sqrt(t))
        P = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return P

    def SMA_indicator(self, SMA_S, SMA_L):
        df = self.get_data()
        df["SMA_S"] = df.price.rolling(SMA_S).mean()
        df["SMA_L"] = df.price.rolling(SMA_L).mean()
        df.dropna(inplace=True)
        return df

    def get_RSI_SMA(self, window=14, SMA_S=20, SMA_L=50):
        df = self.get_data()
        price_series = df['price'].squeeze()
        df['RSI'] = ta.momentum.RSIIndicator(price_series, window=window).rsi() # pyright: ignore[reportAttributeAccessIssue]
        df["SMA_S"] = df.price.rolling(SMA_S).mean()
        df["SMA_L"] = df.price.rolling(SMA_L).mean()
        df.dropna(inplace=True)
        return df

    def plot_charts(self):
        df = self.get_RSI_SMA()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Price + SMA plot
        ax1.plot(df.index, df["price"], label="Price")
        ax1.plot(df.index, df["SMA_S"], color='green',
                 label=f'SMA_S ({self.get_RSI_SMA.__defaults__[1]})') # pyright: ignore[reportOptionalSubscript]
        ax1.plot(df.index, df["SMA_L"], color='red',
                 label=f'SMA_L ({self.get_RSI_SMA.__defaults__[2]})') # pyright: ignore[reportOptionalSubscript]
        ax1.set_title("Price Chart with SMAs")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        ax1.legend()

        # RSI plot
        ax2.plot(df.index, df["RSI"], color='orange')
        ax2.set_title("RSI")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("RSI")
        ax2.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_Scholes_Call(self):
        df = self.get_data()
        plt.plot(df.index,
                 self.black_scholes_call(57900, df["price"], 0.15, 0.05, 8 / 252))
        plt.title("Black-Scholes Call Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Call Price")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    def fetch_option_data(self, symbol='BANKNIFTY', expiry='20251125', strike=57900, right='C'):
        """Fetch 1-min option data and add as a column to main DataFrame."""

        # Ensure IB connection
        if not self.ib.isConnected():
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            print("Connected:", self.ib.isConnected())
            time.sleep(1)

        # Get main data if not already fetched
        if self.data is None:
            self.get_RSI_SMA()

        # Define contract
        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='NSE',
            currency='INR'
        )

        # Request option data
        data = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',       # 1 day for multiple bars
            barSizeSetting='2 min',  # 1-minute bars
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )

        # If the request returned no data object, bail out early
        if not data:
            print(f"⚠️ No data returned for {symbol} {strike}{right}.")
            return self.data

        options_df = util.df(data)

        # Guard against util.df returning None or an empty DataFrame
        if options_df is None or options_df.empty:
            print(f"⚠️ No data returned for {symbol} {strike}{right}.")
            return self.data

        # Ensure expected column exists before accessing it
        if "close" not in options_df.columns:
            print(f"⚠️ 'close' column not present in option data for {symbol} {strike}{right}.")
            return self.data

        return self.data


# ----- Printing ------- # 
if __name__ == "__main__":
    backtester = Black_Scholes_Backtest("^NSEBANK", "2025-11-10", "2025-11-11")
    
    # Fetch price + SMA + RSI
    df = backtester.get_RSI_SMA()
    print(75 * "-")
    print(df.head())
    print(75 * "-")

    # Black-Scholes call price
    call_price = backtester.black_scholes_call(57900, df["price"], 0.15, 0.05, 15 / 252)
    print("\nBlack-Scholes Call Price:")
    print(call_price.head())

    # Plot charts
    backtester.plot_charts()
    backtester.plot_Scholes_Call()

    # Fetch option data and add to df
    df = backtester.fetch_option_data(symbol='BANKNIFTY', strike=57900, right='C')



    self.data["Black_call"] = backtester.black_scholes_call(
    57900, self.data["price"], 0.15, 0.05, 8 / 252
)

# Fetch IB option data separately
options_df = backtester.fetch_option_data(symbol='BANKNIFTY', strike=57900, right='C')

# Plot both on the same chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=self.data, x=self.data.index, y="Black_call", label="Black-Scholes Call", color="green")
sns.lineplot(data=options_df, x="date", y="close", label="Actual Option Price", color="blue")
plt.title("Black-Scholes vs Actual Option Price")
plt.xlabel("Time")
plt.ylabel("Price (INR)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()