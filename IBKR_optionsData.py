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
            

    def fetch_option_data(self, symbol='BANKNIFTY', expiry='20251125', strike=57900, right='C', bar_size='1 min'):
        # Use the existing IBKR connection from the instance
        ib = self.ib
        if not ib.isConnected():
            print("Connecting to IBKR...")
            ib.connect('127.0.0.1', 7497, clientId=1)
            print("Connected:", ib.isConnected())
            time.sleep(1)  # Give IBKR a moment to establish connection

        # Define contract
        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='NSE',
            currency='INR'
        )

        # Fetch historical data
        data = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting=bar_size,
            whatToShow='MIDPOINT',
            useRTH=True
        )

        # Convert to DataFrame
        df = util.df(data)

        if df is None or df.empty:
            print(f"⚠️ No data returned for {symbol} {strike}{right}.")
            return None

        print("Data fetched successfully:\n", df.head())

        # Plot using seaborn
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="date", y="close", label=f"{symbol} {strike}{right}", color="blue")
        plt.title(f"{symbol} {strike}{right} Option - Historical Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return df
            
    def brownian_motion(self, S0=59200, mu=0.05, sigma=0.2, T=1/252, steps=390): 
        dt = T / steps 
        Z = np.random.normal(0, 1, steps)
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z 
        prices = S0 * np.exp(np.cumsum(increments))
        return prices 


    def monte_carlo_simulations(self, S0=59200, mu=0.10, sigma=0.30, T=1/252, steps=390, N=100000):
        final_prices = np.zeros(N)

        for i in range(N):
            path = self.brownian_motion(S0=S0, mu=mu, sigma=sigma, T=T, steps=steps)
            final_prices[i] = path[-1]

        # Print stats inside the same function
        print(f"Expected final price: {np.mean(final_prices)}")
        print(f"Expected standard deviation: {np.std(final_prices)}")

        # Plot inside the same function
        plt.hist(final_prices, bins=50)
        plt.title("BankNifty — 1-Day Monte Carlo GBM")
        plt.xlabel("Final Price")
        plt.ylabel("Frequency")
        plt.show()

        return final_prices




# ----- Printing ------- #
if __name__ == "__main__":
    backtester = Black_Scholes_Backtest("^NSEBANK", "2025-11-13", "2025-11-14")

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

    GBM_prices = backtester.brownian_motion()
    print(GBM_prices)
    plt.plot(GBM_prices)
    plt.title("Brownian Motion")
    plt.show() 

    print(75 * "-")
    


