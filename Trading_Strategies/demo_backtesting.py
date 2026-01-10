#!/usr/bin/env python3
"""
Demonstration script for the three backtesting methodologies:
1. Iterative Backtesting
2. Vectorized Backtesting
3. Monte Carlo Backtesting with GBM
"""

import pandas as pd
import numpy as np
from Quant_backtesting_methods import (
    IterativeBacktester,
    VectorizedBacktester,
    GBMMonteCarloBacktester,
    sma_strategy_example,
    mean_reversion_strategy_example
)


def demo_iterative_backtesting():
    """Demonstrate iterative backtesting with SMA strategy."""
    print("\n" + "="*60)
    print("ITERATIVE BACKTESTING DEMONSTRATION")
    print("="*60)

    # Create backtester instance
    backtester = IterativeBacktester(
        symbol="AAPL",
        start_date="2022-01-01",
        end_date="2023-01-01",
        initial_balance=10000
    )

    # Set risk management parameters
    backtester.set_risk_parameters(stop_loss_pct=0.05, take_profit_pct=0.10)

    # Run backtest with SMA strategy
    backtester.backtest_strategy(sma_strategy_example, sma_short=20, sma_long=50)

    # Print results
    backtester.print_performance_summary()

    # Plot results
    try:
        backtester.plot_results()
    except Exception as e:
        print(f"Plotting error: {e}")


def demo_vectorized_backtesting():
    """Demonstrate vectorized backtesting with multiple strategies."""
    print("\n" + "="*60)
    print("VECTORIZED BACKTESTING DEMONSTRATION")
    print("="*60)

    # Create backtester instance
    backtester = VectorizedBacktester(
        symbol="MSFT",
        start_date="2022-01-01",
        end_date="2023-01-01",
        initial_balance=10000
    )

    # Add technical indicators
    backtester.add_technical_indicators()

    # Test SMA crossover strategy
    print("\nSMA Crossover Strategy:")
    sma_metrics = backtester.backtest_sma_crossover(sma_short=20, sma_long=50)
    backtester.print_performance_summary(sma_metrics)

    # Test RSI mean reversion strategy
    print("\nRSI Mean Reversion Strategy:")
    rsi_metrics = backtester.backtest_rsi_mean_reversion(rsi_period=14, overbought=70, oversold=30)
    backtester.print_performance_summary(rsi_metrics)

    # Test momentum strategy
    print("\nMomentum Strategy:")
    mom_metrics = backtester.backtest_momentum(lookback_period=20)
    backtester.print_performance_summary(mom_metrics)

    # Plot results
    try:
        backtester.plot_results()
    except Exception as e:
        print(f"Plotting error: {e}")


def demo_monte_carlo_backtesting():
    """Demonstrate Monte Carlo backtesting with GBM."""
    print("\n" + "="*60)
    print("MONTE CARLO BACKTESTING DEMONSTRATION")
    print("="*60)

    # Create Monte Carlo backtester
    mc_backtester = GBMMonteCarloBacktester(
        symbol="GOOGL",
        start_date="2022-01-01",
        end_date="2023-01-01",
        initial_balance=10000,
        n_simulations=1000  # Reduced for faster demo
    )

    # Test SMA crossover strategy
    print("\nMonte Carlo SMA Crossover Strategy:")
    mc_results = mc_backtester.backtest_strategy_monte_carlo(
        mc_backtester.sma_crossover_mc,
        sma_short=20,
        sma_long=50
    )
    mc_backtester.print_mc_summary()

    # Test mean reversion strategy
    print("\nMonte Carlo Mean Reversion Strategy:")
    mc_results_mr = mc_backtester.backtest_strategy_monte_carlo(
        mc_backtester.mean_reversion_mc,
        lookback=20,
        threshold=2.0
    )
    mc_backtester.print_mc_summary()

    # Compare with buy and hold
    print("\nMonte Carlo Buy and Hold Strategy:")
    mc_results_bh = mc_backtester.backtest_strategy_monte_carlo(
        mc_backtester.buy_and_hold_strategy
    )
    mc_backtester.print_mc_summary()

    # Plot results
    try:
        mc_backtester.plot_mc_results(n_paths_to_plot=20)
    except Exception as e:
        print(f"Plotting error: {e}")


def compare_strategies():
    """Compare performance across different backtesting methodologies."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON ACROSS BACKTESTING METHODOLOGIES")
    print("="*80)

    symbol = "TSLA"
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Iterative Backtesting
    print("\n1. ITERATIVE BACKTESTING RESULTS:")
    iter_bt = IterativeBacktester(symbol, start_date, end_date)
    iter_bt.backtest_strategy(sma_strategy_example, sma_short=20, sma_long=50)
    iter_metrics = iter_bt.performance_metrics

    # Vectorized Backtesting
    print("\n2. VECTORIZED BACKTESTING RESULTS:")
    vec_bt = VectorizedBacktester(symbol, start_date, end_date)
    vec_metrics = vec_bt.backtest_sma_crossover(sma_short=20, sma_long=50)

    # Monte Carlo Backtesting
    print("\n3. MONTE CARLO BACKTESTING RESULTS:")
    mc_bt = GBMMonteCarloBacktester(symbol, start_date, end_date, n_simulations=500)
    mc_results = mc_bt.backtest_strategy_monte_carlo(mc_bt.sma_crossover_mc, sma_short=20, sma_long=50)

    # Comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Iterative':<12} {'Vectorized':<12} {'Monte Carlo':<12}")
    print("-" * 80)

    metrics_to_compare = [
        ('Total Return', 'total_return', 'total_return', 'mean_return'),
        ('Sharpe Ratio', 'sharpe_ratio', 'sharpe_ratio', 'sharpe_ratio'),
        ('Max Drawdown', 'max_drawdown', 'max_drawdown', None),
        ('Win Rate', 'win_rate', 'win_rate', 'win_rate'),
    ]

    for metric_name, iter_key, vec_key, mc_key in metrics_to_compare:
        iter_val = iter_metrics.get(iter_key, 'N/A')
        vec_val = vec_metrics.get(vec_key, 'N/A')
        mc_val = mc_results.get(mc_key, 'N/A') if mc_key else 'N/A'

        # Format values
        if isinstance(iter_val, (int, float)) and not np.isnan(iter_val):
            iter_str = f"{iter_val:.3f}" if iter_key.endswith('ratio') or iter_key == 'win_rate' else f"{iter_val:.2%}"
        else:
            iter_str = str(iter_val)

        if isinstance(vec_val, (int, float)) and not np.isnan(vec_val):
            vec_str = f"{vec_val:.3f}" if vec_key.endswith('ratio') or vec_key == 'win_rate' else f"{vec_val:.2%}"
        else:
            vec_str = str(vec_val)

        if isinstance(mc_val, (int, float)) and not np.isnan(mc_val):
            mc_str = f"{mc_val:.3f}" if mc_key.endswith('ratio') or mc_key == 'win_rate' else f"{mc_val:.2%}"
        else:
            mc_str = str(mc_val)

        print(f"{metric_name:<25} {iter_str:<12} {vec_str:<12} {mc_str:<12}")

    print("="*80)


if __name__ == "__main__":
    print("QUANTITATIVE BACKTESTING FRAMEWORK DEMONSTRATION")
    print("This demo showcases three rigorous backtesting methodologies:")
    print("1. Iterative Backtesting - Event-driven, bar-by-bar processing")
    print("2. Vectorized Backtesting - Fast pandas/numpy operations")
    print("3. Monte Carlo Backtesting - GBM-based scenario analysis")

    try:
        # Run demonstrations
        demo_iterative_backtesting()
        demo_vectorized_backtesting()
        demo_monte_carlo_backtesting()
        compare_strategies()

        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key Features Demonstrated:")
        print("• Event-driven iterative backtesting with risk management")
        print("• Vectorized operations for fast strategy testing")
        print("• Monte Carlo simulations using Geometric Brownian Motion")
        print("• Comprehensive performance metrics and visualization")
        print("• Cross-methodology performance comparison")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
