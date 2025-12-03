import numpy as np 
import sympy as smp 
import scipy as sp 
import matplotlib.pyplot as plt 
from IterativeBase_Options import * # type ignore
import pandas as pd


# Although this strategy was made for day trading, it often fails for intra day as it assumes random walk and normal distribution, which is true over longer periods of time, but not volatile option markets. I instead suggest that this is used for finding ranges and trends, mathamatically and technically. Then use the ranges as a map to where the market might be headed.  

class GBM_MonteCarlo_Strategy(IterativeBase):
    
    def simulate_forward_GBM(self, S0, mu=0.0, sigma=0.2, steps=10, paths=50, dt=1/252):
        """
        Simulate GBM price paths forward from current price S0
        Returns array shape (paths, steps)
        """
        Z = np.random.normal(size=(paths, steps))
        increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
        log_S = np.cumsum(increments, axis=1)
        S = S0 * np.exp(log_S)
        return S

    def compute_RSI(self, prices, period=14):
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()
        RS = roll_up / roll_down
        RSI = 100 - (100 / (1 + RS))
        return RSI
    
    def run_strategy(self, horizon=5, paths=100, amount_per_trade=None, mu=0.0, sigma=0.2,
                     prob_threshold=0.6, sma_period=20, rsi_period=14):
        """
        Monte Carlo GBM predictive strategy with SMA + RSI filters.
        """
        df = self.data.copy()
        n = len(df)
        
        # Compute SMA and RSI
        df['SMA'] = df['price'].rolling(sma_period).mean()
        df['RSI'] = self.compute_RSI(df['price'], period=rsi_period)
        
        if amount_per_trade is None:
            amount_per_trade = self.initial_balance * 0.1  # 10% default

        for bar in range(max(sma_period, rsi_period), n - horizon):
            S0 = df.price.iloc[bar]
            sim_prices = self.simulate_forward_GBM(S0, mu=mu, sigma=sigma, steps=horizon, paths=paths)
            prob_up = (sim_prices[:, -1] > S0).mean()
            max_units = int(amount_per_trade / S0)
            
            # Get SMA and RSI values at current bar
            sma = df.SMA.iloc[bar]
            rsi = df.RSI.iloc[bar]

            # ---------------------------------------
            # LONG SIGNAL
            # ---------------------------------------
            if prob_up >= prob_threshold and S0 > sma and rsi < 70 and self.units == 0:
                if max_units > 0:
                    self.buy_instrument(bar, units=max_units)
            
            # ---------------------------------------
            # SHORT / EXIT SIGNAL
            # ---------------------------------------
            elif prob_up <= (1 - prob_threshold) and (S0 < sma or rsi > 70) and self.units > 0:
                self.sell_instrument(bar, units=self.units)

            # Optional: print NAV every 50 bars
            if bar % 50 == 0:
                self.print_current_nav(bar)
        
        # Close final position
        if self.units != 0:
            self.close_pos(n - 1)
        
        print("=== Monte Carlo + SMA + RSI Strategy Completed ===")
        return self.current_balance

# ------------------- Usage -------------------

ticker = GBM_MonteCarlo_Strategy("BANKNIFTY", "20251230", "59500", "C", 28000, True)
ticker.get_data()  # Load real option data

final_balance = ticker.run_strategy(
    horizon=5,
    paths=100,
    amount_per_trade=5000,
    mu=0.0,
    sigma=0.2,
    prob_threshold=0.6,
    sma_period=10,
    rsi_period=14
)

print("Final Balance:", round(final_balance))
