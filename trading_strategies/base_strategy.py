import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import json


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    This provides the framework that our CI/CD pipeline will use.
    """

    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.backtest_results = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on market data.
        Must be implemented by each strategy.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
        """
        pass

    def backtest(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Run backtest for this strategy on given symbol and date range.
        This is what our CI/CD pipeline will call automatically.
        """
        print(f"🔄 Backtesting {self.name} on {symbol} from {start_date} to {end_date}")

        # Download market data
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
        except Exception as e:
            return {"error": f"Failed to download data: {str(e)}"}

        # Generate trading signals
        signals = self.generate_signals(data)

        # Calculate returns
        returns = self.calculate_returns(data, signals)

        # Calculate performance metrics
        metrics = self.calculate_metrics(returns, data)

        # Store results
        self.backtest_results = {
            "strategy_name": self.name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "parameters": self.parameters,
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"],
            "benchmark_return": metrics["benchmark_return"],
            "outperformance": metrics["outperformance"]
        }

        print(f"✅ Backtest complete: {metrics['annual_return']:.2%} annual return")
        return self.backtest_results

    def calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns based on signals"""

        # Merge data with signals
        df = data.copy()
        df['signal'] = signals['signal']

        # Calculate daily returns
        df['market_return'] = df['Close'].pct_change()

        # Strategy returns: only get market return when we have a position
        df['position'] = df['signal'].shift(1)  # Use previous day's signal
        df['strategy_return'] = df['position'] * df['market_return']

        # Handle missing values
        df = df.fillna(0)

        return df

    def calculate_metrics(self, returns_df: pd.DataFrame, original_data: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""

        strategy_returns = returns_df['strategy_return'].fillna(0)
        market_returns = returns_df['market_return'].fillna(0)

        # Basic calculations
        total_strategy_return = (1 + strategy_returns).prod() - 1
        total_market_return = (1 + market_returns).prod() - 1

        # Annualized returns
        days = len(returns_df)
        years = days / 252  # 252 trading days per year
        annual_strategy_return = (1 + total_strategy_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = strategy_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_strategy_return / volatility if volatility != 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        positive_returns = strategy_returns[strategy_returns > 0]
        win_rate = len(positive_returns) / len(strategy_returns[strategy_returns != 0]) if len(
            strategy_returns[strategy_returns != 0]) > 0 else 0

        # Total trades (number of signal changes)
        signals = returns_df['position'].fillna(0)
        total_trades = len(signals[signals.diff() != 0]) - 1  # -1 to exclude first position

        return {
            "total_return": total_strategy_return,
            "annual_return": annual_strategy_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "benchmark_return": total_market_return,
            "outperformance": annual_strategy_return - (total_market_return ** (1 / years) - 1) if years > 0 else 0
        }

    def save_results(self, filepath: str):
        """Save backtest results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.backtest_results, f, indent=2, default=str)
        print(f"📊 Results saved to {filepath}")

    def get_optimization_params(self) -> Dict:
        """
        Return parameters that can be optimized by AI.
        Override this in your strategy to specify which parameters to optimize.
        """
        return {}

    def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        self.parameters.update(new_params)
        print(f"🔧 Updated parameters: {new_params}")

    def is_valid_strategy(self) -> Tuple[bool, str]:
        """
        Validate if strategy meets minimum requirements.
        Our CI/CD pipeline will use this to auto-reject bad strategies.
        """
        if not self.backtest_results:
            return False, "No backtest results available"

        # Define minimum requirements
        min_sharpe = 0.5
        max_drawdown_limit = -0.3  # -30%
        min_trades = 10

        # Check requirements
        if self.backtest_results["sharpe_ratio"] < min_sharpe:
            return False, f"Sharpe ratio {self.backtest_results['sharpe_ratio']:.2f} below minimum {min_sharpe}"

        if self.backtest_results["max_drawdown"] < max_drawdown_limit:
            return False, f"Max drawdown {self.backtest_results['max_drawdown']:.2%} exceeds limit {max_drawdown_limit:.2%}"

        if self.backtest_results["total_trades"] < min_trades:
            return False, f"Only {self.backtest_results['total_trades']} trades, minimum {min_trades} required"

        return True, "Strategy passes validation"