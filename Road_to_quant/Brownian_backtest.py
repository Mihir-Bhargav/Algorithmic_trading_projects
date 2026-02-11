"""
Monte Carlo Backtesting Module

This module implements various Monte Carlo simulation methods for backtesting
trading strategies to assess their robustness and statistical significance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BacktestResults:
    """Container for backtest results"""
    equity_curve: np.ndarray
    returns: np.ndarray
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


class MonteCarloBacktester:
    """
    Monte Carlo Backtester with multiple randomization methods:
    1. Trade Sequence Randomization
    2. Bootstrap Resampling
    3. Return Distribution Sampling
    4. Block Bootstrap (preserves temporal correlation)
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        
    def calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics from returns series"""
        equity_curve = self.initial_capital * (1 + returns).cumprod()
        
        # Total return
        total_return = (equity_curve[-1] / self.initial_capital - 1) * 100
        
        # Sharpe ratio (annualized, assuming daily returns)
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve # pyright: ignore[reportReturnType] GPT suggested using Union[]
        }
    
    def trade_randomization(self, 
                           trades: List[float], 
                           n_simulations: int = 1000) -> pd.DataFrame:
        """
        Method 1: Randomize the sequence of trades
        
        This method shuffles the order of trades while keeping the same
        trade outcomes to test if the strategy's performance is order-dependent.
        """
        results = []
        original_metrics = self.calculate_metrics(np.array(trades))
        
        for i in range(n_simulations):
            # Shuffle trades
            shuffled_trades = np.random.permutation(trades)
            metrics = self.calculate_metrics(shuffled_trades)
            results.append(metrics)
        
        return pd.DataFrame(results), original_metrics # pyright: ignore[reportReturnType]
    
    def bootstrap_resampling(self,
                            returns: np.ndarray,
                            n_simulations: int = 1000,
                            sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Method 2: Bootstrap resampling of returns
        
        Randomly sample returns with replacement to generate new equity curves.
        """
        if sample_size is None:
            sample_size = len(returns)
            
        results = []
        
        for i in range(n_simulations):
            # Sample with replacement
            sampled_returns = np.random.choice(returns, size=sample_size, replace=True)
            metrics = self.calculate_metrics(sampled_returns)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def parametric_simulation(self,
                            returns: np.ndarray,
                            n_simulations: int = 1000,
                            distribution: str = 'normal') -> pd.DataFrame:
        """
        Method 3: Generate returns from fitted distribution
        
        Fit a distribution to returns and generate new samples from it.
        """
        results = []
        n_periods = len(returns)
        
        if distribution == 'normal':
            mu = returns.mean()
            sigma = returns.std()
            
            for i in range(n_simulations):
                simulated_returns = np.random.normal(mu, sigma, n_periods)
                metrics = self.calculate_metrics(simulated_returns)
                results.append(metrics)
        
        elif distribution == 't':
            # Fit t-distribution (captures fat tails)
            from scipy import stats
            params = stats.t.fit(returns)
            
            for i in range(n_simulations):
                simulated_returns = stats.t.rvs(*params, size=n_periods)
                metrics = self.calculate_metrics(simulated_returns) # pyright: ignore[reportArgumentType]
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def block_bootstrap(self,
                       returns: np.ndarray,
                       block_size: int = 20,
                       n_simulations: int = 1000) -> pd.DataFrame:
        """
        Method 4: Block bootstrap to preserve temporal correlation
        
        Sample blocks of consecutive returns to maintain autocorrelation structure.
        """
        results = []
        n_periods = len(returns)
        n_blocks = int(np.ceil(n_periods / block_size))
        
        for i in range(n_simulations):
            # Create blocks
            blocks = []
            for j in range(n_blocks):
                start_idx = np.random.randint(0, max(1, len(returns) - block_size + 1))
                end_idx = min(start_idx + block_size, len(returns))
                blocks.append(returns[start_idx:end_idx])
            
            # Concatenate blocks and trim to original length
            sampled_returns = np.concatenate(blocks)[:n_periods]
            metrics = self.calculate_metrics(sampled_returns)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def run_full_monte_carlo(self,
                            returns: np.ndarray,
                            n_simulations: int = 1000,
                            methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Run all Monte Carlo methods and return results
        """
        if methods is None:
            methods = ['bootstrap', 'parametric', 'block_bootstrap']
        
        results = {}
        original_metrics = self.calculate_metrics(returns)
        results['original'] = original_metrics
        
        if 'bootstrap' in methods:
            results['bootstrap'] = self.bootstrap_resampling(returns, n_simulations)
        
        if 'parametric' in methods:
            results['parametric'] = self.parametric_simulation(returns, n_simulations)
        
        if 'block_bootstrap' in methods:
            results['block_bootstrap'] = self.block_bootstrap(returns, n_simulations=n_simulations)
        
        return results
    
    def analyze_results(self, 
                       monte_carlo_results: pd.DataFrame,
                       original_metric: float,
                       metric_name: str = 'total_return') -> Dict:
        """
        Analyze Monte Carlo results to assess statistical significance
        """
        simulated_values = monte_carlo_results[metric_name]
        
        # Percentile rank
        percentile = (simulated_values < original_metric).sum() / len(simulated_values) * 100
        
        # Confidence intervals
        ci_95 = np.percentile(simulated_values, [2.5, 97.5])
        ci_90 = np.percentile(simulated_values, [5, 95])
        
        return {
            'original_value': original_metric,
            'mean_simulated': simulated_values.mean(),
            'std_simulated': simulated_values.std(),
            'percentile_rank': percentile,
            'ci_95': ci_95,
            'ci_90': ci_90,
            'min': simulated_values.min(),
            'max': simulated_values.max()
        }
    
    def plot_results(self,
                    monte_carlo_results: pd.DataFrame,
                    original_metrics: Dict[str, float],
                    metric_name: str = 'total_return',
                    title: str = 'Monte Carlo Simulation Results'):
        """
        Plot Monte Carlo simulation results with original performance
        """
        plt.style.use('dark_background')
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, color='white')
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Color palette for histograms
        colors = ["#80DEEF", "#C9F376", "#71F06A", "#EBA148"]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Histogram of simulated values with themed color
            ax.hist(monte_carlo_results[metric], bins=50, alpha=0.7, 
                   color=colors[idx], edgecolor='white', linewidth=0.5)
            
            # Original value line
            original_value = original_metrics[metric]
            ax.axvline(original_value, color='#FF4444', linestyle='--', 
                      linewidth=2.5, label=f'Original: {original_value:.2f}')
            
            # Mean of simulations
            mean_value = monte_carlo_results[metric].mean()
            ax.axvline(mean_value, color="#23989C", linestyle='--',
                      linewidth=2.5, label=f'MC Mean: {mean_value:.2f}')
            
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.2, color='gray')
        
        plt.tight_layout()
        return fig
    
    def plot_equity_curves(self,
                          monte_carlo_results: pd.DataFrame,
                          original_equity: np.ndarray,
                          n_sample_curves: int = 100,
                          color: str = '#00D9FF'):
        """
        Plot sample equity curves from Monte Carlo simulation
        
        Args:
            monte_carlo_results: DataFrame with Monte Carlo results
            original_equity: Original strategy equity curve
            n_sample_curves: Number of sample curves to plot
            color: Color for the simulated curves (default: cyan)
        """
        plt.style.use('dark_background')
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sample simulated curves
        sample_indices = np.random.choice(len(monte_carlo_results), 
                                         min(n_sample_curves, len(monte_carlo_results)),
                                         replace=False)
        
        for idx in sample_indices:
            equity = monte_carlo_results.iloc[idx]['equity_curve']
            ax.plot(equity, alpha=0.5, color=color, linewidth=0.5)
        
        # Plot original equity curve
        ax.plot(original_equity, color='#FF4444', linewidth=2.5, label='Original Strategy', zorder=100)
        
        ax.set_xlabel('Time Period', fontsize=11)
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.set_title('Monte Carlo Equity Curves', fontsize=13, color='white')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.5, color='gray')
        
        return fig


def example_usage():
    """Example demonstrating how to use the Monte Carlo Backtester"""
    
    # Generate sample returns (you would use actual strategy returns)
    np.random.seed(42)
    n_trades = 252  # One year of daily trades
    
    # Simulate a strategy with positive drift and some wins/losses
    returns = np.random.normal(0.001, 0.02, n_trades)  # Mean 0.1% daily return
    
    # Initialize backtester
    backtester = MonteCarloBacktester(initial_capital=10000)
    
    # Run Monte Carlo simulations
    print("Running Monte Carlo Simulations...")
    results = backtester.run_full_monte_carlo(returns, n_simulations=1000)
    
    # Analyze results
    print("\n" + "="*60)
    print("MONTE CARLO BACKTEST ANALYSIS")
    print("="*60)
    
    original_metrics = results['original']
    print(f"\nOriginal Strategy Performance:")
    print(f"Total Return: {original_metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {original_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {original_metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {original_metrics['win_rate']:.2f}%")
    
    for method_name, mc_results in results.items():
        if method_name == 'original':
            continue
            
        print(f"\n{method_name.upper()} METHOD:")
        print("-" * 60)
        
        analysis = backtester.analyze_results(
            mc_results, 
            original_metrics['total_return'], # pyright: ignore[reportArgumentType]
            'total_return'
        )
        
        print(f"Mean Simulated Return: {analysis['mean_simulated']:.2f}%")
        print(f"Std Dev: {analysis['std_simulated']:.2f}%")
        print(f"Percentile Rank: {analysis['percentile_rank']:.1f}%")
        print(f"95% CI: [{analysis['ci_95'][0]:.2f}%, {analysis['ci_95'][1]:.2f}%]")
        
        # Interpretation
        if analysis['percentile_rank'] > 95:
            print("✓ Strategy performance is statistically significant (top 5%)")
        elif analysis['percentile_rank'] < 5:
            print("⚠ Strategy performance is unusually poor")
        else:
            print("○ Strategy performance is within normal range")
    
    # Generate plots
    print("\nGenerating plots...")
    backtester.plot_results(results['bootstrap'], original_metrics, # pyright: ignore[reportArgumentType]
                           title='Bootstrap Monte Carlo Results')
    
    backtester.plot_equity_curves(results['bootstrap'], 
                                 original_metrics['equity_curve'], # pyright: ignore[reportArgumentType]
                                 n_sample_curves=100)
    
    plt.show()
    
    return backtester, results


if __name__ == "__main__":
    backtester, results = example_usage()