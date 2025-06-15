import pandas as pd
import numpy as np
from base_strategy import BaseStrategy
from typing import Dict


class MovingAverageStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy

    Buy when short-term MA crosses above long-term MA
    Sell when short-term MA crosses below long-term MA

    This is what a developer would commit to GitLab, and our
    CI/CD pipeline would automatically tests and optimize it!
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the Moving Average strategy

        Args:
            short_window: Period for short-term moving average
            long_window: Period for long-term moving average
        """
        parameters = {
            "short_window": short_window,
            "long_window": long_window
        }
        super().__init__("Moving Average Crossover", parameters)

        # Store parameters as instance variables for easy access
        self.short_window = short_window
        self.long_window = long_window

        print(f"🤖 Created Moving Average Strategy: {short_window}/{long_window}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on moving average crossover

        Returns:
            DataFrame with 'signal' column:
            1 = Buy signal
            -1 = Sell signal
            0 = Hold/No position
        """

        df = data.copy()

        # Calculate moving averages
        df[f'MA_{self.short_window}'] = df['Close'].rolling(window=self.short_window).mean()
        df[f'MA_{self.long_window}'] = df['Close'].rolling(window=self.long_window).mean()

        # Initialize signal column
        df['signal'] = 0

        # Generate signals
        # Buy when short MA crosses above long MA
        df.loc[df[f'MA_{self.short_window}'] > df[f'MA_{self.long_window}'], 'signal'] = 1

        # Sell when short MA crosses below long MA
        df.loc[df[f'MA_{self.short_window}'] < df[f'MA_{self.long_window}'], 'signal'] = -1

        # Only keep actual crossover points (not continuous signals)
        df['signal_change'] = df['signal'].diff()

        # Clean up signals - only act on changes
        final_signals = df.copy()
        final_signals['signal'] = 0

        # Mark buy signals (when signal changes from 0/-1 to 1)
        buy_signals = (df['signal'] == 1) & (df['signal_change'] != 0)
        final_signals.loc[buy_signals, 'signal'] = 1

        # Mark sell signals (when signal changes from 1 to -1)
        sell_signals = (df['signal'] == -1) & (df['signal_change'] != 0)
        final_signals.loc[sell_signals, 'signal'] = -1

        # Forward fill positions (hold position until next signal)
        final_signals['position'] = final_signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        final_signals['signal'] = final_signals['position']

        return final_signals[['signal']]

    def get_optimization_params(self) -> Dict:
        """
        Define which parameters our AI can optimize

        Our CI/CD pipeline will use this to automatically
        find the best parameters for this strategy!
        """
        return {
            "short_window": {
                "type": "integer",
                "min": 5,
                "max": 50,
                "current": self.short_window
            },
            "long_window": {
                "type": "integer",
                "min": 20,
                "max": 200,
                "current": self.long_window
            }
        }

    def update_parameters(self, new_params: Dict):
        """Update strategy parameters and validate them"""
        super().update_parameters(new_params)

        # Update instance variables
        if "short_window" in new_params:
            self.short_window = new_params["short_window"]
        if "long_window" in new_params:
            self.long_window = new_params["long_window"]

        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError(f"Short window ({self.short_window}) must be less than long window ({self.long_window})")

        print(f"✅ Updated to MA {self.short_window}/{self.long_window}")


# Example usage and testing
if __name__ == "__main__":
    # This is what gets executed when our CI/CD pipeline tests the strategy

    print("🚀 Testing Moving Average Strategy")
    print("=" * 50)

    # Create strategy instance
    strategy = MovingAverageStrategy(short_window=20, long_window=50)

    # Run backtest on Apple stock for the last year
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    results = strategy.backtest(
        symbol="AAPL",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    # Display results
    if "error" not in results:
        print(f"\n📊 Backtest Results for {results['strategy_name']}:")
        print(f"   📈 Annual Return: {results['annual_return']:.2%}")
        print(f"   📉 Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"   ⚡ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   🎯 Win Rate: {results['win_rate']:.2%}")
        print(f"   📊 Total Trades: {results['total_trades']}")
        print(f"   🏆 Outperformance: {results['outperformance']:.2%}")

        # Check if strategy passes validation
        is_valid, message = strategy.is_valid_strategy()
        if is_valid:
            print(f"   ✅ Status: {message}")
        else:
            print(f"   ❌ Status: {message}")

        # Save results for our dashboard
        strategy.save_results("../reports/ma_strategy_results.json")

    else:
        print(f"❌ Error: {results['error']}")

    print("\n🔧 Optimization Parameters:")
    opt_params = strategy.get_optimization_params()
    for param, config in opt_params.items():
        print(f"   {param}: {config['current']} (range: {config['min']}-{config['max']})")

    print("\n🎯 Ready for CI/CD Integration!")
