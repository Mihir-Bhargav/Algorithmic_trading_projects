# Quantitative Backtesting Framework

A comprehensive collection of three rigorous backtesting methodologies for quantitative trading strategy development.

## Files

### 1. `Quant_backtesting_methods.py`
Main implementation file containing three backtesting classes:

- **IterativeBacktester**: Event-driven, bar-by-bar backtesting with advanced risk management
- **VectorizedBacktester**: Fast vectorized operations using pandas/numpy for high-speed strategy testing
- **GBMMonteCarloBacktester**: Monte Carlo simulation using Geometric Brownian Motion for scenario analysis

### 2. `demo_backtesting.py`
Demonstration script showcasing all three backtesting methodologies with real market data examples.

## Features

### IterativeBacktester
- Event-driven processing for complex trading logic
- Risk management (stop-loss, take-profit)
- Detailed trade logging
- Position sizing and cost accounting
- Real-time portfolio tracking

### VectorizedBacktester
- Multiple built-in strategies (SMA crossover, RSI mean reversion, momentum)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Parameter optimization capabilities
- Instant computation across entire datasets

### GBMMonteCarloBacktester
- Parameter estimation from historical data
- Multiple strategy implementations for MC testing
- Risk metrics (VaR, CVaR, Sharpe ratio)
- Probability analysis across scenarios
- Stress testing under different market conditions

## Requirements

```python
pandas
numpy
yfinance
matplotlib
scipy
```

Install with:
```bash
pip install pandas numpy yfinance matplotlib scipy
```

## Quick Start

```python
from Quant_backtesting_methods import (
    IterativeBacktester,
    VectorizedBacktester,
    GBMMonteCarloBacktester
)

# Iterative Backtesting
iterative_bt = IterativeBacktester("AAPL", "2022-01-01", "2023-01-01")
iterative_bt.backtest_strategy(your_strategy_function)
iterative_bt.print_performance_summary()

# Vectorized Backtesting
vectorized_bt = VectorizedBacktester("MSFT", "2022-01-01", "2023-01-01")
metrics = vectorized_bt.backtest_sma_crossover(sma_short=20, sma_long=50)
vectorized_bt.print_performance_summary(metrics)

# Monte Carlo Backtesting
mc_bt = GBMMonteCarloBacktester("GOOGL", "2022-01-01", "2023-01-01", n_simulations=1000)
results = mc_bt.backtest_strategy_monte_carlo(mc_bt.sma_crossover_mc)
mc_bt.print_mc_summary()
```

## Performance Metrics

All backtesters provide comprehensive metrics including:
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Volatility
- Alpha (vs Buy & Hold)
- Value at Risk (VaR) - Monte Carlo only
- Conditional VaR - Monte Carlo only

## Run Demo

```bash
cd Trading_Strategies
python demo_backtesting.py
```

## Notes

- Data is fetched using yfinance from Yahoo Finance
- All classes include realistic cost modeling (bid-ask spreads, trading costs)
- Results can be visualized using built-in plotting functions
- The framework is designed for professional quantitative research applications

