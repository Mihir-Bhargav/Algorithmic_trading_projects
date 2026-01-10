import pandas as pd 
import numpy as np 
import yfinance as yf 
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
plt.style.use("seaborn-v0_8")


class IterativeBacktester:
    """
    Event-driven backtesting framework that processes data bar by bar,
    allowing for complex trading logic, position sizing, and risk management.
    """

    def __init__(self, symbol, start_date, end_date, initial_balance=10000, spread_cost=0.0005):
        """
        Initialize the iterative backtester.

        Parameters:
        -----------
        symbol : str
            Trading instrument symbol
        start_date : str
            Start date for backtesting (YYYY-MM-DD)
        end_date : str
            End date for backtesting (YYYY-MM-DD)
        initial_balance : float
            Initial capital for trading
        spread_cost : float
            Bid-ask spread cost as fraction of price
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.spread_cost = spread_cost

        # Portfolio state
        self.current_balance = initial_balance
        self.units = 0
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.trades = 0
        self.trade_log = []

        # Performance tracking
        self.portfolio_values = []
        self.returns = []

        # Risk management
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.max_drawdown = 0
        self.peak_value = initial_balance

        self.get_data()

    def get_data(self):
        """Fetch and prepare historical price data."""
        try:
            raw = yf.download(self.symbol, self.start_date, self.end_date)
            raw = raw.copy().dropna()
            raw = raw.loc[self.start_date:self.end_date]
            raw.rename(columns={"Close": "price", "High": "high", "Low": "low",
                              "Open": "open", "Volume": "volume"}, inplace=True)
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
            raw["spread"] = self.spread_cost * (raw["high"] - raw["low"])
            raw = raw.dropna().copy()
            self.data = raw
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            self.data = pd.DataFrame()

    def set_risk_parameters(self, stop_loss_pct=None, take_profit_pct=None):
        """Set risk management parameters."""
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def check_risk_management(self, bar, entry_price):
        """Check if risk management conditions are met."""
        current_price = self.data.iloc[bar]["price"]

        if self.stop_loss_pct and self.position != 0:
            if self.position == 1:  # Long position
                if current_price <= entry_price * (1 - self.stop_loss_pct):
                    return "stop_loss"
            else:  # Short position
                if current_price >= entry_price * (1 + self.stop_loss_pct):
                    return "stop_loss"

        if self.take_profit_pct and self.position != 0:
            if self.position == 1:  # Long position
                if current_price >= entry_price * (1 + self.take_profit_pct):
                    return "take_profit"
            else:  # Short position
                if current_price <= entry_price * (1 - self.take_profit_pct):
                    return "take_profit"

        return None

    def execute_trade(self, bar, side, units=None, amount=None):
        """
        Execute a trade with proper cost accounting.

        Parameters:
        -----------
        bar : int
            Current bar index
        side : str
            'buy' or 'sell'
        units : int, optional
            Number of units to trade
        amount : float, optional
            Dollar amount to trade
        """
        price = self.data.iloc[bar]["price"]
        spread = self.data.iloc[bar]["spread"]

        if side == "buy":
            # Buy at ask price (price + half spread)
            execution_price = price + spread/2
        else:
            # Sell at bid price (price - half spread)
            execution_price = price - spread/2

        if amount is not None:
            units = amount / execution_price
        elif units is None:
            raise ValueError("Must specify either units or amount")

        if side == "buy":
            self.current_balance -= units * execution_price
            self.units += units
            trade_type = "BUY"
        else:
            self.current_balance += units * execution_price
            self.units -= units
            trade_type = "SELL"

        self.trades += 1

        # Log trade
        trade_record = {
            'bar': bar,
            'date': self.data.index[bar],
            'type': trade_type,
            'price': execution_price,
            'units': units,
            'balance': self.current_balance,
            'position': self.units
        }
        self.trade_log.append(trade_record)

        return execution_price

    def update_portfolio_value(self, bar):
        """Update portfolio value and track performance metrics."""
        price = self.data.iloc[bar]["price"]
        if hasattr(price, 'iloc'):
            price = price.iloc[0]
        portfolio_value = self.current_balance + self.units * price
        self.portfolio_values.append(portfolio_value)

        # Calculate returns
        if len(self.portfolio_values) > 1:
            ret = np.log(self.portfolio_values[-1] / self.portfolio_values[-2])
            self.returns.append(ret)

            # Track drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def backtest_strategy(self, strategy_func, **kwargs):
        """
        Execute backtest using a user-defined strategy function.

        Parameters:
        -----------
        strategy_func : callable
            Function that takes (self, bar) and returns trading signal
        **kwargs : dict
            Additional parameters for strategy function
        """
        print(f"Starting iterative backtest for {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial balance: ${self.initial_balance:,.2f}")
        print("-" * 60)

        entry_price = None

        for bar in range(len(self.data)):
            current_price = self.data.iloc[bar]["price"]

            # Update portfolio value at the beginning of each bar
            self.update_portfolio_value(bar)

            # Check risk management conditions
            if entry_price is not None:
                risk_signal = self.check_risk_management(bar, entry_price)
                if risk_signal:
                    if self.position == 1:
                        self.execute_trade(bar, "sell", units=abs(self.units))
                    elif self.position == -1:
                        self.execute_trade(bar, "buy", units=abs(self.units))

                    self.position = 0
                    entry_price = None
                    continue

            # Get trading signal from strategy
            signal = strategy_func(self, bar, **kwargs)

            # Execute trades based on signal
            if signal == "buy" and self.position <= 0:
                if self.position == -1:  # Close short position first
                    self.execute_trade(bar, "buy", units=abs(self.units))

                # Go long
                amount = self.current_balance
                if amount > 0:
                    self.execute_trade(bar, "buy", amount=amount)
                    self.position = 1
                    entry_price = current_price

            elif signal == "sell" and self.position >= 0:
                if self.position == 1:  # Close long position first
                    self.execute_trade(bar, "sell", units=abs(self.units))

                # Go short
                amount = self.current_balance
                if amount > 0:
                    self.execute_trade(bar, "sell", amount=amount)
                    self.position = -1
                    entry_price = current_price

            elif signal == "close" and self.position != 0:
                if self.position == 1:
                    self.execute_trade(bar, "sell", units=abs(self.units))
                elif self.position == -1:
                    self.execute_trade(bar, "buy", units=abs(self.units))
                self.position = 0
                entry_price = None

        # Close any remaining position
        if self.position != 0:
            if self.position == 1:
                self.execute_trade(bar, "sell", units=abs(self.units))
            elif self.position == -1:
                self.execute_trade(bar, "buy", units=abs(self.units))

        self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_values:
            return {}

        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance

        # Calculate Sharpe ratio (assuming daily returns)
        if len(self.returns) > 1:
            sharpe_ratio = np.sqrt(252) * np.mean(self.returns) / np.std(self.returns)
        else:
            sharpe_ratio = 0

        # Calculate Sortino ratio
        if len(self.returns) > 1:
            negative_returns = [r for r in self.returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                sortino_ratio = np.sqrt(252) * np.mean(self.returns) / downside_std
            else:
                sortino_ratio = float('inf')
        else:
            sortino_ratio = 0

        # Calculate win rate
        winning_trades = sum(1 for trade in self.trade_log[1:] if trade['balance'] > self.trade_log[0]['balance'])
        win_rate = winning_trades / max(1, len(self.trade_log) - 1)

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.trades,
            'final_balance': final_value
        }

        self.performance_metrics = metrics
        return metrics

    def plot_results(self):
        """Plot backtesting results."""
        if not hasattr(self, 'portfolio_values') or not self.portfolio_values:
            print("No results to plot. Run backtest first.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        ax1.plot(self.data.index[:len(self.portfolio_values)], self.portfolio_values)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)

        # Asset price
        ax2.plot(self.data.index, self.data['price'], color='blue', alpha=0.7)
        ax2.set_title(f'{self.symbol} Price')
        ax2.set_ylabel('Price ($)')
        ax2.grid(True)

        # Returns distribution
        if self.returns:
            ax3.hist(self.returns, bins=50, alpha=0.7, color='green')
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Daily Return')
            ax3.grid(True)

        # Trade log visualization
        if self.trade_log:
            trade_dates = [trade['date'] for trade in self.trade_log]
            trade_balances = [trade['balance'] for trade in self.trade_log]
            ax4.plot(trade_dates, trade_balances, 'ro-', markersize=3)
            ax4.set_title('Balance After Each Trade')
            ax4.set_ylabel('Balance ($)')
            ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self):
        """Print detailed performance summary."""
        if not hasattr(self, 'performance_metrics'):
            print("Run backtest first to calculate metrics.")
            return

        metrics = self.performance_metrics

        print("\n" + "="*60)
        print("BACKTESTING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Strategy: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print("="*60)


class VectorizedBacktester:
    """
    Vectorized backtesting framework using pandas/numpy operations
    for fast computation of trading strategies across entire datasets.
    """

    def __init__(self, symbol, start_date, end_date, initial_balance=10000, spread_cost=0.0005):
        """
        Initialize the vectorized backtester.

        Parameters:
        -----------
        symbol : str
            Trading instrument symbol
        start_date : str
            Start date for backtesting (YYYY-MM-DD)
        end_date : str
            End date for backtesting (YYYY-MM-DD)
        initial_balance : float
            Initial capital for trading
        spread_cost : float
            Bid-ask spread cost as fraction of price
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.spread_cost = spread_cost
        self.get_data()

    def get_data(self):
        """Fetch and prepare historical price data."""
        try:
            raw = yf.download(self.symbol, self.start_date, self.end_date)
            raw = raw.copy().dropna()
            raw = raw.loc[self.start_date:self.end_date]
            raw.rename(columns={"Close": "price", "High": "high", "Low": "low",
                              "Open": "open", "Volume": "volume"}, inplace=True)
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
            raw["spread"] = self.spread_cost * (raw["high"] - raw["low"])
            raw = raw.dropna().copy()
            self.data = raw
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            self.data = pd.DataFrame()

    def add_technical_indicators(self):
        """Add common technical indicators to the dataset."""
        df = self.data.copy()

        # Moving averages
        df['SMA_20'] = df['price'].rolling(20).mean()
        df['SMA_50'] = df['price'].rolling(50).mean()
        df['EMA_20'] = df['price'].ewm(span=20).mean()

        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        df['BB_middle'] = df['price'].rolling(20).mean()
        df['BB_std'] = df['price'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

        # Volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['price'].shift(1)).abs()
        low_close = (df['low'] - df['price'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        self.data = df.dropna().copy()

    def backtest_sma_crossover(self, sma_short=20, sma_long=50, position_size=1.0):
        """
        Backtest SMA crossover strategy using vectorized operations.

        Parameters:
        -----------
        sma_short : int
            Short-term SMA period
        sma_long : int
            Long-term SMA period
        position_size : float
            Position size as fraction of balance
        """
        df = self.data.copy()

        # Generate signals
        df['SMA_short'] = df['price'].rolling(sma_short).mean()
        df['SMA_long'] = df['price'].rolling(sma_long).mean()
        df['signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)
        df['position'] = df['signal'].shift(1)  # Use previous day's signal

        # Account for spread costs
        df['spread_cost'] = np.where(df['position'] != df['position'].shift(1),
                                   df['spread'], 0)

        # Calculate strategy returns
        df['strategy_returns'] = df['position'] * df['returns'] - df['spread_cost']
        df['cum_strategy_returns'] = df['strategy_returns'].cumsum()
        df['strategy_price'] = self.initial_balance * np.exp(df['cum_strategy_returns'])

        # Calculate buy and hold returns
        df['bh_returns'] = df['returns'].cumsum()
        df['bh_price'] = self.initial_balance * np.exp(df['bh_returns'])

        # Calculate drawdowns
        df['strategy_peak'] = df['strategy_price'].cummax()
        df['strategy_drawdown'] = (df['strategy_peak'] - df['strategy_price']) / df['strategy_peak']

        self.results = df.dropna().copy()
        return self.calculate_performance_metrics(df)

    def backtest_rsi_mean_reversion(self, rsi_period=14, overbought=70, oversold=30):
        """
        Backtest RSI-based mean reversion strategy.

        Parameters:
        -----------
        rsi_period : int
            RSI calculation period
        overbought : float
            RSI overbought threshold
        oversold : float
            RSI oversold threshold
        """
        df = self.data.copy()

        # Calculate RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Generate signals
        df['signal'] = 0
        df.loc[df['RSI'] > overbought, 'signal'] = -1  # Short when overbought
        df.loc[df['RSI'] < oversold, 'signal'] = 1     # Long when oversold

        df['position'] = df['signal'].shift(1)

        # Account for spread costs
        df['spread_cost'] = np.where(df['position'] != df['position'].shift(1),
                                   df['spread'], 0)

        # Calculate strategy returns
        df['strategy_returns'] = df['position'] * df['returns'] - df['spread_cost']
        df['cum_strategy_returns'] = df['strategy_returns'].cumsum()
        df['strategy_price'] = self.initial_balance * np.exp(df['cum_strategy_returns'])

        self.results = df.dropna().copy()
        return self.calculate_performance_metrics(df)

    def backtest_momentum(self, lookback_period=20, position_size=1.0):
        """
        Backtest momentum strategy.

        Parameters:
        -----------
        lookback_period : int
            Period to calculate momentum
        position_size : float
            Position size as fraction of balance
        """
        df = self.data.copy()

        # Calculate momentum
        df['momentum'] = df['price'] / df['price'].shift(lookback_period) - 1

        # Generate signals
        df['signal'] = np.where(df['momentum'] > 0, 1, -1)
        df['position'] = df['signal'].shift(1)

        # Account for spread costs
        df['spread_cost'] = np.where(df['position'] != df['position'].shift(1),
                                   df['spread'], 0)

        # Calculate strategy returns
        df['strategy_returns'] = df['position'] * df['returns'] - df['spread_cost']
        df['cum_strategy_returns'] = df['strategy_returns'].cumsum()
        df['strategy_price'] = self.initial_balance * np.exp(df['cum_strategy_returns'])

        self.results = df.dropna().copy()
        return self.calculate_performance_metrics(df)

    def calculate_performance_metrics(self, df):
        """Calculate comprehensive performance metrics."""
        if df.empty:
            return {}

        # Basic returns
        total_return = (df['strategy_price'].iloc[-1] - self.initial_balance) / self.initial_balance
        bh_return = (df['bh_price'].iloc[-1] - self.initial_balance) / self.initial_balance

        # Risk-adjusted metrics
        daily_returns = df['strategy_returns'].dropna()
        if len(daily_returns) > 1:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

            # Sortino ratio
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                sortino_ratio = np.sqrt(252) * daily_returns.mean() / negative_returns.std()
            else:
                sortino_ratio = float('inf')
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Drawdown analysis
        max_drawdown = df['strategy_drawdown'].max()

        # Win rate
        winning_days = (daily_returns > 0).sum()
        win_rate = winning_days / len(daily_returns) if len(daily_returns) > 0 else 0

        # Additional metrics
        volatility = daily_returns.std() * np.sqrt(252)
        alpha = total_return - bh_return

        metrics = {
            'total_return': total_return,
            'buy_hold_return': bh_return,
            'alpha': alpha,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'total_days': len(df),
            'final_balance': df['strategy_price'].iloc[-1]
        }

        return metrics

    def plot_results(self):
        """Plot backtesting results."""
        if not hasattr(self, 'results') or self.results.empty:
            print("No results to plot. Run backtest first.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value comparison
        ax1.plot(self.results.index, self.results['strategy_price'], label='Strategy')
        ax1.plot(self.results.index, self.results['bh_price'], label='Buy & Hold', alpha=0.7)
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)

        # Asset price with signals
        ax2.plot(self.results.index, self.results['price'], color='blue', alpha=0.7)
        if 'SMA_short' in self.results.columns:
            ax2.plot(self.results.index, self.results['SMA_short'], color='green', alpha=0.7, label='SMA Short')
            ax2.plot(self.results.index, self.results['SMA_long'], color='red', alpha=0.7, label='SMA Long')
        ax2.set_title(f'{self.symbol} Price')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True)

        # Returns distribution
        ax3.hist(self.results['strategy_returns'], bins=50, alpha=0.7, color='green')
        ax3.set_title('Strategy Returns Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.grid(True)

        # Drawdown
        ax4.fill_between(self.results.index, 0, -self.results['strategy_drawdown'], color='red', alpha=0.3)
        ax4.set_title('Strategy Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self, metrics=None):
        """Print detailed performance summary."""
        if metrics is None and hasattr(self, 'results'):
            metrics = self.calculate_performance_metrics(self.results)
        elif metrics is None:
            print("No metrics available. Run backtest first.")
            return

        print("\n" + "="*60)
        print("VECTORIZED BACKTESTING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Strategy: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"Strategy Return: {metrics['total_return']:.2%}")
        print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
        print(f"Alpha: {metrics['alpha']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Total Trading Days: {metrics['total_days']}")
        print("="*60)


class GBMMonteCarloBacktester:
    """
    Monte Carlo backtesting framework using Geometric Brownian Motion
    to simulate multiple price paths and evaluate strategy performance
    across different market scenarios.
    """

    def __init__(self, symbol, start_date, end_date, initial_balance=10000,
                 n_simulations=1000, spread_cost=0.0005):
        """
        Initialize the GBM Monte Carlo backtester.

        Parameters:
        -----------
        symbol : str
            Trading instrument symbol
        start_date : str
            Start date for backtesting (YYYY-MM-DD)
        end_date : str
            End date for backtesting (YYYY-MM-DD)
        initial_balance : float
            Initial capital for trading
        n_simulations : int
            Number of Monte Carlo simulations
        spread_cost : float
            Bid-ask spread cost as fraction of price
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.n_simulations = n_simulations
        self.spread_cost = spread_cost

        self.get_data()
        self.estimate_parameters()

    def get_data(self):
        """Fetch and prepare historical price data."""
        try:
            raw = yf.download(self.symbol, self.start_date, self.end_date)
            raw = raw.copy().dropna()
            raw = raw.loc[self.start_date:self.end_date]
            raw.rename(columns={"Close": "price"}, inplace=True)
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
            raw = raw.dropna().copy()
            self.data = raw
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            self.data = pd.DataFrame()

    def estimate_parameters(self):
        """Estimate GBM parameters from historical data."""
        if self.data.empty:
            self.mu_daily = 0
            self.sigma_daily = 0.01
            self.S0 = 100
            return

        returns = self.data['returns'].dropna()
        self.mu_daily = returns.mean()
        self.sigma_daily = returns.std()
        self.S0 = self.data['price'].iloc[0]

        # Annualize parameters
        self.mu_annual = self.mu_daily * 252
        self.sigma_annual = self.sigma_daily * np.sqrt(252)

        print(f"Estimated parameters for {self.symbol}:")
        print(f"Daily drift (μ): {self.mu_daily:.6f}")
        print(f"Daily volatility (σ): {self.sigma_daily:.6f}")
        print(f"Annual drift: {self.mu_annual:.4f}")
        print(f"Annual volatility: {self.sigma_annual:.4f}")

    def simulate_gbm_path(self, T_days):
        """
        Simulate a single GBM price path.

        Parameters:
        -----------
        T_days : int
            Number of days to simulate

        Returns:
        --------
        numpy.ndarray
            Simulated price path
        """
        dt = 1  # Daily time step
        drift = (self.mu_daily - 0.5 * self.sigma_daily**2) * dt
        diffusion = self.sigma_daily * np.sqrt(dt) * np.random.normal(0, 1, T_days)

        # Generate log returns
        log_returns = drift + diffusion

        # Generate price path
        prices = self.S0 * np.exp(np.cumsum(log_returns))

        return prices

    def simulate_multiple_paths(self, T_days):
        """
        Simulate multiple GBM price paths.

        Parameters:
        -----------
        T_days : int
            Number of days to simulate

        Returns:
        --------
        numpy.ndarray
            Array of shape (n_simulations, T_days) with price paths
        """
        paths = np.zeros((self.n_simulations, T_days))

        for i in range(self.n_simulations):
            paths[i] = self.simulate_gbm_path(T_days)

        return paths

    def backtest_strategy_monte_carlo(self, strategy_func, T_days=None, **kwargs):
        """
        Backtest strategy using Monte Carlo simulations.

        Parameters:
        -----------
        strategy_func : callable
            Strategy function that takes price path and returns portfolio values
        T_days : int, optional
            Trading days to simulate (default: length of historical data)
        **kwargs : dict
            Additional parameters for strategy function

        Returns:
        --------
        dict
            Monte Carlo simulation results
        """
        if T_days is None:
            T_days = len(self.data)

        print(f"Running {self.n_simulations} Monte Carlo simulations...")
        print(f"Simulating {T_days} trading days...")

        # Simulate price paths
        price_paths = self.simulate_multiple_paths(T_days)

        # Run strategy on each simulated path
        portfolio_paths = np.zeros((self.n_simulations, T_days))
        strategy_returns = []

        for i in range(self.n_simulations):
            path_values = strategy_func(price_paths[i], **kwargs)
            portfolio_paths[i] = path_values

            # Calculate total return for this simulation
            if len(path_values) > 1:
                total_return = (path_values[-1] - self.initial_balance) / self.initial_balance
                strategy_returns.append(total_return)

        strategy_returns = np.array(strategy_returns)

        # Calculate statistics
        results = {
            'portfolio_paths': portfolio_paths,
            'price_paths': price_paths,
            'strategy_returns': strategy_returns,
            'mean_return': np.mean(strategy_returns),
            'median_return': np.median(strategy_returns),
            'std_return': np.std(strategy_returns),
            'var_95': np.percentile(strategy_returns, 5),  # 95% VaR
            'cvar_95': strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)].mean(),
            'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0,
            'win_rate': np.mean(strategy_returns > 0),
            'max_return': np.max(strategy_returns),
            'min_return': np.min(strategy_returns),
            'prob_positive': np.mean(strategy_returns > 0),
            'prob_loss_10pct': np.mean(strategy_returns < -0.10),
            'prob_gain_20pct': np.mean(strategy_returns > 0.20)
        }

        self.mc_results = results
        return results

    def buy_and_hold_strategy(self, price_path, spread_cost=0.0005):
        """
        Buy and hold strategy for Monte Carlo testing.

        Parameters:
        -----------
        price_path : numpy.ndarray
            Simulated price path
        spread_cost : float
            Trading cost

        Returns:
        --------
        numpy.ndarray
            Portfolio values over time
        """
        shares = self.initial_balance / (price_path[0] * (1 + spread_cost))
        portfolio_values = shares * price_path * (1 - spread_cost)
        return portfolio_values

    def sma_crossover_mc(self, price_path, sma_short=20, sma_long=50, spread_cost=0.0005):
        """
        SMA crossover strategy for Monte Carlo testing.

        Parameters:
        -----------
        price_path : numpy.ndarray
            Simulated price path
        sma_short : int
            Short SMA period
        sma_long : int
            Long SMA period
        spread_cost : float
            Trading cost

        Returns:
        --------
        numpy.ndarray
            Portfolio values over time
        """
        portfolio_values = []
        position = 0  # 1 for long, -1 for short, 0 for cash
        cash = self.initial_balance

        for i in range(len(price_path)):
            current_price = price_path[i]

            # Calculate SMAs (using available data up to current point)
            if i >= sma_long - 1:
                short_sma = np.mean(price_path[max(0, i-sma_short+1):i+1])
                long_sma = np.mean(price_path[max(0, i-sma_long+1):i+1])

                # Generate signal
                if short_sma > long_sma and position <= 0:
                    # Buy signal
                    if position == -1:  # Close short position
                        cash += abs(position) * current_price * (1 - spread_cost)
                        position = 0

                    # Go long
                    shares_to_buy = cash / (current_price * (1 + spread_cost))
                    cash -= shares_to_buy * current_price * (1 + spread_cost)
                    position = shares_to_buy

                elif short_sma < long_sma and position >= 0:
                    # Sell signal
                    if position == 1:  # Close long position
                        cash += position * current_price * (1 - spread_cost)
                        position = 0

                    # Go short
                    shares_to_sell = cash / (current_price * (1 + spread_cost))
                    cash += shares_to_sell * current_price * (1 - spread_cost)
                    position = -shares_to_sell

            # Calculate current portfolio value
            if position > 0:  # Long position
                portfolio_value = cash + position * current_price
            elif position < 0:  # Short position
                portfolio_value = cash + position * current_price  # position is negative
            else:  # Cash position
                portfolio_value = cash

            portfolio_values.append(portfolio_value)

        return np.array(portfolio_values)

    def mean_reversion_mc(self, price_path, lookback=20, threshold=2.0, spread_cost=0.0005):
        """
        Mean reversion strategy for Monte Carlo testing.

        Parameters:
        -----------
        price_path : numpy.ndarray
            Simulated price path
        lookback : int
            Lookback period for mean calculation
        threshold : float
            Standard deviation threshold for signals
        spread_cost : float
            Trading cost

        Returns:
        --------
        numpy.ndarray
            Portfolio values over time
        """
        portfolio_values = []
        position = 0
        cash = self.initial_balance

        for i in range(len(price_path)):
            current_price = price_path[i]

            if i >= lookback:
                # Calculate rolling mean and std
                window = price_path[i-lookback:i]
                mean_price = np.mean(window)
                std_price = np.std(window)

                # Calculate z-score
                z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

                # Generate signals
                if z_score < -threshold and position <= 0:
                    # Buy signal (price below mean)
                    if position == -1:
                        cash += abs(position) * current_price * (1 - spread_cost)
                        position = 0

                    # Go long
                    shares_to_buy = cash / (current_price * (1 + spread_cost))
                    cash -= shares_to_buy * current_price * (1 + spread_cost)
                    position = shares_to_buy

                elif z_score > threshold and position >= 0:
                    # Sell signal (price above mean)
                    if position == 1:
                        cash += position * current_price * (1 - spread_cost)
                        position = 0

                    # Go short
                    shares_to_sell = cash / (current_price * (1 + spread_cost))
                    cash += shares_to_sell * current_price * (1 - spread_cost)
                    position = -shares_to_sell

            # Calculate portfolio value
            if position > 0:
                portfolio_value = cash + position * current_price
            elif position < 0:
                portfolio_value = cash + position * current_price
            else:
                portfolio_value = cash

            portfolio_values.append(portfolio_value)

        return np.array(portfolio_values)

    def plot_mc_results(self, n_paths_to_plot=50):
        """Plot Monte Carlo simulation results."""
        if not hasattr(self, 'mc_results'):
            print("No Monte Carlo results to plot. Run backtest first.")
            return

        results = self.mc_results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value paths
        paths_to_plot = min(n_paths_to_plot, self.n_simulations)
        indices = np.random.choice(self.n_simulations, paths_to_plot, replace=False)

        for idx in indices:
            ax1.plot(results['portfolio_paths'][idx], alpha=0.3, color='blue', linewidth=0.5)

        ax1.set_title(f'Portfolio Value Paths ({paths_to_plot} simulations)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)

        # Returns distribution
        ax2.hist(results['strategy_returns'], bins=50, alpha=0.7, color='green', density=True)
        ax2.axvline(results['mean_return'], color='red', linestyle='--',
                   label=f'Mean: {results["mean_return"]:.2%}')
        ax2.axvline(results['var_95'], color='orange', linestyle='--',
                   label=f'95% VaR: {results["var_95"]:.2%}')
        ax2.set_title('Strategy Returns Distribution')
        ax2.set_xlabel('Total Return')
        ax2.legend()
        ax2.grid(True)

        # Price paths
        for idx in indices[:10]:  # Plot fewer price paths
            ax3.plot(results['price_paths'][idx], alpha=0.3, color='purple', linewidth=0.5)

        ax3.set_title('Simulated Price Paths')
        ax3.set_ylabel('Price ($)')
        ax3.grid(True)

        # Summary statistics text
        ax4.text(0.1, 0.9, f'Mean Return: {results["mean_return"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.8, f'Median Return: {results["median_return"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.7, f'Std Dev: {results["std_return"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.6, f'95% VaR: {results["var_95"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.5, f'Win Rate: {results["win_rate"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.4, f'Sharpe Ratio: {results["sharpe_ratio"]:.3f}', fontsize=10)
        ax4.text(0.1, 0.3, f'P(Return > 20%): {results["prob_gain_20pct"]:.2%}', fontsize=10)
        ax4.text(0.1, 0.2, f'P(Loss > 10%): {results["prob_loss_10pct"]:.2%}', fontsize=10)
        ax4.set_title('Monte Carlo Statistics')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

    def print_mc_summary(self):
        """Print Monte Carlo simulation summary."""
        if not hasattr(self, 'mc_results'):
            print("No Monte Carlo results available. Run backtest first.")
            return

        results = self.mc_results

        print("\n" + "="*60)
        print("MONTE CARLO BACKTESTING SUMMARY")
        print("="*60)
        print(f"Strategy: {self.symbol}")
        print(f"Simulations: {self.n_simulations}")
        print(f"Historical Period: {self.start_date} to {self.end_date}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print()

        print("RETURN STATISTICS:")
        print(f"Mean Return: {results['mean_return']:.2%}")
        print(f"Median Return: {results['median_return']:.2%}")
        print(f"Standard Deviation: {results['std_return']:.2%}")
        print(f"Best Case: {results['max_return']:.2%}")
        print(f"Worst Case: {results['min_return']:.2%}")
        print()

        print("RISK METRICS:")
        print(f"95% Value at Risk (VaR): {results['var_95']:.2%}")
        print(f"95% Conditional VaR: {results['cvar_95']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print()

        print("PROBABILITY ANALYSIS:")
        print(f"Probability of Positive Return: {results['prob_positive']:.2%}")
        print(f"Probability of Return > 20%: {results['prob_gain_20pct']:.2%}")
        print(f"Probability of Loss > 10%: {results['prob_loss_10pct']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print("="*60)


# Example usage functions for demonstration
def sma_strategy_example(backtester, bar, sma_short=20, sma_long=50):
    """Example SMA crossover strategy for iterative backtesting."""
    if bar < sma_long:
        return None

    # Calculate SMAs
    short_sma = backtester.data['price'].iloc[bar-sma_short+1:bar+1].mean()
    if hasattr(short_sma, 'iloc'):
        short_sma = short_sma.iloc[0]
    long_sma = backtester.data['price'].iloc[bar-sma_long+1:bar+1].mean()
    if hasattr(long_sma, 'iloc'):
        long_sma = long_sma.iloc[0]

    if short_sma > long_sma:
        return "buy"
    elif short_sma < long_sma:
        return "sell"

    return None


def mean_reversion_strategy_example(backtester, bar, lookback=20, threshold=2.0):
    """Example mean reversion strategy for iterative backtesting."""
    if bar < lookback:
        return None

    # Calculate z-score
    window = backtester.data['price'].iloc[bar-lookback:bar]
    mean_price = window.mean()
    if hasattr(mean_price, 'iloc'):
        mean_price = mean_price.iloc[0]
    std_price = window.std()
    if hasattr(std_price, 'iloc'):
        std_price = std_price.iloc[0]
    current_price = backtester.data['price'].iloc[bar]
    if hasattr(current_price, 'iloc'):
        current_price = current_price.iloc[0]

    if std_price > 0:
        z_score = (current_price - mean_price) / std_price

        if z_score < -threshold:
            return "buy"
        elif z_score > threshold:
            return "sell"

    return None


# Demo function to run all three backtesters
def run_comprehensive_backtest(symbol="AAPL", start_date="2020-01-01", end_date="2023-01-01"):
    """
    Demonstrate all three backtesting approaches.
    """
    print("COMPREHENSIVE BACKTESTING DEMONSTRATION")
    print("="*60)

    # 1. Iterative Backtesting
    print("\n1. ITERATIVE BACKTESTING")
    iterative_bt = IterativeBacktester(symbol, start_date, end_date)
    iterative_bt.set_risk_parameters(stop_loss_pct=0.05, take_profit_pct=0.10)
    iterative_bt.backtest_strategy(sma_strategy_example, sma_short=20, sma_long=50)
    iterative_bt.print_performance_summary()

    # 2. Vectorized Backtesting
    print("\n2. VECTORIZED BACKTESTING")
    vectorized_bt = VectorizedBacktester(symbol, start_date, end_date)
    metrics = vectorized_bt.backtest_sma_crossover(sma_short=20, sma_long=50)
    vectorized_bt.print_performance_summary(metrics)

    # 3. Monte Carlo Backtesting
    print("\n3. MONTE CARLO BACKTESTING")
    mc_bt = GBMMonteCarloBacktester(symbol, start_date, end_date, n_simulations=500)
    mc_results = mc_bt.backtest_strategy_monte_carlo(mc_bt.sma_crossover_mc,
                                                   sma_short=20, sma_long=50)
    mc_bt.print_mc_summary()

    return iterative_bt, vectorized_bt, mc_bt


