import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from IterativeBase_Options import IterativeBase

class SchrodingerStrategy(IterativeBase):
    
    def compute_RSI(self, prices, period=14):
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()
        RS = roll_up / roll_down
        RSI = 100 - (100 / (1 + RS))
        return RSI
    
    def build_transition_matrix(self, n_bins, p_up=0.4, p_down=0.4):
        """Create a simple tri-diagonal transition matrix"""
        T = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            if i > 0:
                T[i, i-1] = p_down
            T[i, i] = 1 - p_up - p_down
            if i < n_bins-1:
                T[i, i+1] = p_up
        return T
    
    def run_strategy(
        self,
        horizon: int = 10,
        amount_per_trade: float | None = None,
        grid_size: int = 10,
        sma_period: int = 20,
        rsi_period: int = 14,
        rsi_long_thresh: float = 65.0,
        rsi_short_thresh: float = 35.0,
        prob_bias_scale: float = 50.0,
        p_stay: float = 0.1,
        cooldown_bars: int = 1,
    ):
        """
        Discrete-time Schrödinger-inspired trading loop (optimized & robust).

        Key points / defaults:
        - horizon: how many bars forward to propagate the transition matrix
        - grid_size: +/- bins around S0 (total bins = 2*grid_size + 1)
        - amount_per_trade: cash used per trade (defaults to 5% of initial balance)
        - sma_period / rsi_period: indicator windows
        - rsi_long_thresh / rsi_short_thresh: relaxed RSI filters
        - prob_bias_scale: scales estimated local drift -> bias in up-probability
        - p_stay: probability to remain in same bin each step (tri-diagonal matrix)
        - cooldown_bars: minimum bars between trades to avoid flip-flopping

        This method:
        - builds a small centered price grid per bar
        - constructs a tri-diagonal stochastic transition matrix T
        - computes T**horizon using matrix power (fast)
        - propagates a delta distribution and reads expected future price
        - applies SMA + RSI + probability-bias filters
        - uses buy_instrument / sell_instrument for execution
        """
        # ---- prepare data and indicators ----
        df = self.data.copy()
        n = len(df)
        if n == 0:
            print("No data available.")
            return self.current_balance

        # ensure indicators exist
        df["SMA"] = df["price"].rolling(sma_period).mean()
        # RSI (simple implementation)
        delta = df["price"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(rsi_period).mean()
        roll_down = down.rolling(rsi_period).mean()
        RS = roll_up / roll_down
        df["RSI"] = 100 - (100 / (1 + RS))

        # defaults
        if amount_per_trade is None:
            amount_per_trade = max(self.initial_balance * 0.05, 1.0)  # default 5% of capital, minimum 1

        last_trade_bar = -999

        # iterate bars (start after indicators warmed up)
        start_bar = max(sma_period, rsi_period)
        for bar in range(start_bar, n - horizon):
            S0 = float(df["price"].iat[bar])
            sma = float(df["SMA"].iat[bar])
            rsi = float(df["RSI"].iat[bar])

            # small safety: if S0 is 0 or nan, skip
            if not np.isfinite(S0) or S0 <= 0:
                continue

            # adaptive grid step: use spread if sensible else % of price
            spread = float(df["spread"].iat[bar]) if "spread" in df.columns else 0.0
            delta_S = max( max(0.01, spread), S0 * 0.002 )  # at least 0.01 or 0.2% of price

            # build centered price grid so S0 is exactly at center index
            n_bins = 2 * grid_size + 1
            center = grid_size
            price_grid = S0 + (np.arange(n_bins) - center) * delta_S

            # estimate short-term drift from recent returns (robust small-sample estimate)
            lookback = max(3, min(20, int(horizon * 3)))
            start_rb = max(1, bar - lookback)
            recent_rets = df["price"].iloc[start_rb:bar].pct_change().dropna()
            drift_est = recent_rets.mean() if len(recent_rets) > 0 else 0.0

            # convert drift estimate -> bias in upward probability
            # scaled so small drifts produce small bias; bound bias to [-0.15, 0.15]
            bias = float(np.clip(drift_est * prob_bias_scale, -0.15, 0.15))

            # base up/down probabilities then normalize with p_stay
            p_up = 0.5 + bias
            p_down = 1.0 - p_up
            # allocate some mass to staying in same bin
            p_up = max(0.0, p_up * (1.0 - p_stay))
            p_down = max(0.0, p_down * (1.0 - p_stay))
            # if numeric rounding pushes total > 1, renormalize
            total = p_up + p_down + p_stay
            if total <= 0:
                p_up = p_down = (1.0 - p_stay) / 2.0
            else:
                p_up /= total
                p_down /= total
                # reassign p_stay so sum = 1
                p_stay = 1.0 - (p_up + p_down)

            # build tri-diagonal transition matrix T efficiently
            # T[i,i] = p_stay ; T[i,i+1] = p_up ; T[i,i-1] = p_down
            T = np.zeros((n_bins, n_bins), dtype=float)
            idx = np.arange(n_bins)
            T[idx, idx] = p_stay
            if n_bins > 1:
                T[idx[:-1], idx[1:]] = p_up  # up moves
                T[idx[1:], idx[:-1]] = p_down  # down moves
            # fix boundaries: at edges there is no up/down outside grid -> reflect mass inward
            T[0, 0] += p_down   # bottom bin cannot go down; keep probability
            T[-1, -1] += p_up   # top bin cannot go up; keep probability

            # fast exponentiation of T: T_power = T ** horizon (matrix power)
            # if horizon is small this is cheap; for larger horizon can use repeated squaring
            try:
                T_power = np.linalg.matrix_power(T, horizon)
            except Exception:
                # fallback: iterative multiplication
                T_power = T.copy()
                for _ in range(1, horizon):
                    T_power = T_power @ T

            # initial distribution — delta at center
            psi0 = np.zeros(n_bins, dtype=float)
            psi0[center] = 1.0

            # propagate
            psi_final = T_power @ psi0  # resulting probability distribution after horizon steps

            # expected future price (probabilities already sum to 1)
            expected_price = float(np.dot(price_grid, psi_final))

            # position sizing: integer units appropriate for options (floor)
            max_units = int(amount_per_trade / max(S0, 1e-6))
            if max_units <= 0:
                # amount_per_trade too small relative to option price -> skip trading
                max_units = 0

            # Enforce a tiny cooldown to avoid immediate flip-flops
            if bar - last_trade_bar <= cooldown_bars:
                continue

            # --- Trading decision: simplified but robust ---
            long_condition = (expected_price > S0) and (rsi < rsi_long_thresh)
            short_condition = (expected_price < S0) and (rsi > rsi_short_thresh)

            # Prefer long-only for options unless you intentionally short (here we exit longs)
            if long_condition and self.units == 0 and max_units > 0:
                # execute buy
                self.buy_instrument(bar, units=max_units)
                last_trade_bar = bar

            elif short_condition and self.units > 0:
                # exit position (sell all)
                self.sell_instrument(bar, units=self.units)
                last_trade_bar = bar

            # periodic logging
            if bar % 50 == 0:
                self.print_current_nav(bar)

        # close final position if still open
        if self.units != 0:
            self.close_pos(n - 1)

        print("=== Schrodinger-inspired strategy completed ===")
        return self.current_balance

ticker = SchrodingerStrategy(
    "BANKNIFTY",     # symbol
    "20251230",      # expiry
    "59500",         # strike
    "P",             # right
    28000,           # initial balance
    True             # live data = True (you already use this style)
)

ticker.get_data()  # Load the option data

final_balance = ticker.run_strategy(
    horizon=10,
    amount_per_trade=5000,
    grid_size=10,
    sma_period=20,
    rsi_period=14,
    rsi_long_thresh=65,
    rsi_short_thresh=35,
    prob_bias_scale=50,
    p_stay=0.1,
    cooldown_bars=1
)

print("Final Balance:", round(final_balance))
