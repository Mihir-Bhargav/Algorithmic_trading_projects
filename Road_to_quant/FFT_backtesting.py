import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy


class BacktestFFT:
    """
    Strategy-agnostic FFT-enhanced backtesting engine
    """

    def __init__(
        self,
        prices: pd.Series,
        signal_fn,
        fft_window=128,
        fft_stride=1,
        entropy_threshold=2.5,
        use_returns=True,
        transaction_cost=0.0005,
        execution_delay=1,
    ):
        self.prices = prices.dropna()
        self.signal_fn = signal_fn
        self.fft_window = fft_window
        self.fft_stride = fft_stride
        self.entropy_threshold = entropy_threshold
        self.use_returns = use_returns
        self.transaction_cost = transaction_cost
        self.execution_delay = execution_delay

        self.results = None

    # --------------------------------------------------
    # FFT diagnostics
    # --------------------------------------------------
    def _spectral_entropy(self, x):
        spec = np.abs(fft(x)) ** 2
        spec = spec[: len(spec) // 2]
        spec /= spec.sum() + 1e-12
        return entropy(spec)

    def fft_entropy_series(self):
        series = (
            self.prices.pct_change().dropna()
            if self.use_returns
            else self.prices
        )

        ent = pd.Series(index=series.index, dtype=float)

        for i in range(self.fft_window, len(series), self.fft_stride):
            window = series.iloc[i - self.fft_window : i].values
            ent.iloc[i] = self._spectral_entropy(window)

        return ent.ffill()

    # --------------------------------------------------
    # Backtest runner
    # --------------------------------------------------
    def run(self):
        prices = self.prices
        returns = prices.pct_change().fillna(0)

        # Strategy signal
        raw_signal = self.signal_fn(prices)
        raw_signal = raw_signal.reindex(prices.index).fillna(0)

        # FFT regime filter
        fft_entropy = self.fft_entropy_series()
        regime_ok = (fft_entropy < self.entropy_threshold).astype(int)

        # Final tradable signal
        signal = raw_signal * regime_ok
        signal = signal.shift(self.execution_delay).fillna(0)

        # PnL
        pnl = signal * returns

        # Transaction costs
        trades = signal.diff().abs()
        pnl -= trades * self.transaction_cost

        equity = (1 + pnl).cumprod()

        self.results = pd.DataFrame({
            "price": prices,
            "signal": signal,
            "fft_entropy": fft_entropy,
            "returns": returns,
            "pnl": pnl,
            "equity": equity
        })

        return self.results

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    def stats(self):
        if self.results is None:
            raise RuntimeError("Run backtest first.")

        pnl = self.results["pnl"]
        equity = self.results["equity"]

        sharpe = np.sqrt(252) * pnl.mean() / (pnl.std() + 1e-12)
        max_dd = (equity / equity.cummax() - 1).min()

        return {
            "Total Return": equity.iloc[-1] - 1,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Trades": self.results["signal"].diff().abs().sum(),
        }


